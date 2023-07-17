from sampler import LossAwareSampler,UniformSampler
INITIAL_LOG_LOSS_SCALE = 20.0
import numpy as np
import torch
import copy
import functools

from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        epochs,
        schedule_sampler=None,
        weight_decay=0.0,
        save_interval=100,
        saved_path="",
        scheduler_factor=0.5
    ):
        self.scheduler = None
        self.model = model
        self.diffusion = diffusion
        self.trainloader = data[0]
        self.testloader = data[1]
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.saved_path = saved_path
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        #self.log_interval = log_interval
        self.save_interval = save_interval
        #self.resume_checkpoint = resume_checkpoint
        self.resume_checkpoint = ""
        #self.use_fp16 = use_fp16
        #self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        #self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.sync_cuda else "cpu"

        self._load_and_sync_parameters()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = lr_scheduler.LinearLR(self.opt, start_factor=1.0, end_factor=scheduler_factor, total_iters=100)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(torch.load(resume_checkpoint, map_location=self.device))

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
            #state_dict = dist_util.load_state_dict(
            #        ema_checkpoint, map_location=dist_util.dev()
            #    )
           #     ema_params = self._state_dict_to_master_params(state_dict)
            ema_params = self._state_dict_to_master_params(torch.load(ema_checkpoint, map_location=self.device))

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(torch.load(opt_checkpoint, map_location=self.device))


    def run_loop(self):
        self.loss_tab = {"loss":[],"mse":[],"vb":[]}
        self.t_loss_tab = {"loss":[],"mse":[],"vb":[]}
        for epoch in range(self.n_epochs):
            
            self.current_loss = {"loss":0,"mse":0,"vb":0}
            self.t_loss = {"loss":0,"mse":0,"vb":0}

            for iteration, data in enumerate(self.trainloader, 0):
                with torch.autocast("cuda") and torch.enable_grad():
                    torch.cuda.empty_cache()
                    inputs, targets = data[0].to(self.device).float().to(self.device).swapaxes(1,2), data[1].to(self.device).swapaxes(1,2).to(self.device).float()
                    loss_dict = self.run_step(targets,inputs)
                    self.current_loss['loss']+= loss_dict["loss"].item()
                    if "mse" in loss_dict:
                        self.current_loss['mse']+= loss_dict["mse"].item()
                    if "vb" in loss_dict:
                        self.current_loss['vb']+= loss_dict["vb"].item()


            self.loss_tab["loss"].append(self.current_loss["loss"]/len(self.trainloader))
            self.loss_tab["mse"].append(self.current_loss["mse"]/len(self.trainloader))
            self.loss_tab["vb"].append(self.current_loss["vb"]/len(self.trainloader))


            print("evaluating...")
            self.test_step()

            
        
            print('[ %d ] loss : %.4f %.4f %.4f %.4f  %.4f %.4f' % (epoch + 1, self.loss_tab["loss"][-1], self.loss_tab["mse"][-1], 
            self.loss_tab["vb"][-1], self.t_loss_tab["loss"][-1],self.t_loss_tab["mse"][-1],self.t_loss_tab["vb"][-1]))




            if self.step % self.save_interval == 0:
                self.plotLosses(self.loss_tab,self.t_loss_tab)
                self.save()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.plotLosses(self.loss_tab,self.t_loss_tab)


    



    def run_step(self, batch, cond):
        self.ddp_model.train()
        loss_dict = self.forward_backward(batch, cond)
        self.optimize_normal()
        return loss_dict

    def forward_backward(self, batch, cond):
        for param in self.model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t,
            cond,
        )
        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        loss_dict = {k: (v * weights).mean() for k, v in losses.items()}
        loss.backward()
        self.scheduler.step()
        return loss_dict

    def test_step(self):
        self.ddp_model.eval()
        for iteration, data in enumerate(self.testloader, 0):
            with torch.autocast("cuda") and torch.enable_grad():
                torch.cuda.empty_cache()
                inputs, targets = data[0].to(self.device).float().swapaxes(1,2), data[1].to(self.device).swapaxes(1,2).to(self.device).float()
                t, weights = self.schedule_sampler.sample(targets.shape[0], self.device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    targets,
                    t,
                    inputs,
                )
                losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                loss_dict = {k: (v * weights).mean() for k, v in losses.items()}

                self.t_loss['loss']+= loss_dict["loss"].item()
                if "mse" in loss_dict:
                    self.t_loss['mse']+= loss_dict["mse"].item()
                if "vb" in loss_dict:
                    self.t_loss['vb']+= loss_dict["vb"].item()

        self.t_loss_tab["loss"].append(self.t_loss["loss"]/(iteration + 1))
        self.t_loss_tab["mse"].append(self.t_loss["mse"]/(iteration + 1))
        self.t_loss_tab["vb"].append(self.t_loss["vb"]/(iteration + 1))
        
    
    def optimize_normal(self):
        #self._log_grad_norm()
        #self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)


    


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.n_epochs
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        print("epoch", self.step + self.resume_step)
        #logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        

    def plotLosses(self,loss_dict,t_loss_dict=None):
        """Loss"""

        fig = plt.figure(dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(range(self.step+1), loss_dict['loss'], label='loss')
        if 'loss' in t_loss_dict:
            ax1.plot(range(self.step+1), t_loss_dict['loss'], label='test loss')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        plt.savefig(self.saved_path+f'loss_epoch_{self.step}.png')
        plt.close()

        fig = plt.figure(dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(range(self.step+1), loss_dict['loss'], label='loss')
        if 'loss' in t_loss_dict:
            ax1.plot(range(self.step+1), t_loss_dict['loss'], label='test loss')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        plt.savefig(self.saved_path+f'loss_epoch_{self.step}.png')
        plt.close()

        """MSE"""
        fig = plt.figure(dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(range(self.step+1), loss_dict['mse'], label='loss')
        if 'mse' in t_loss_dict:
            ax1.plot(range(self.step+1), t_loss_dict['mse'], label='test loss')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        plt.savefig(self.saved_path+f'mse_epoch_{self.step}.png')
        plt.close()

        """vb"""
        fig = plt.figure(dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(range(self.step+1), loss_dict['vb'], label='loss')
        if 'vb' in t_loss_dict:
            ax1.plot(range(self.step+1), t_loss_dict['vb'], label='test loss')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        plt.savefig(self.saved_path+f'vb_epoch_{self.step}.png')
        plt.close()
        





    def save(self):
        torch.save(self.ddp_model.state_dict(), f'{self.saved_path}epoch_{self.step}.pt')
        #torch.save(opt.state_dict(),f'{self.saved_path}optepoch_{self.step}.pt')

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if not main_checkpoint:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def update_ema(target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.
        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)


if __name__ == "__main__":
    print("All Good")
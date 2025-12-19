from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger
from utils import *
from models.SINGLEModel import SINGLEModel
from sklearn.utils.class_weight import compute_class_weight
import os, wandb
from sklearn.metrics import confusion_matrix
from pid_lagrangian import PIDLambdaController

class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.problem = self.args.problem
        self.penalty_factor = args.penalty_factor
        self.pid_lambda_controller = None

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": []}
        if args.tb_logger:
            self.tb_logger = TbLogger(self.log_path)
        else:
            self.tb_logger = None
        self.wandb_logger = args.wandb_logger

        # Main Components
        self.envs = get_env(self.args.problem)  # a list of envs classes (different problems), remember to initialize it!
        self.model = SINGLEModel(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.model)

        # Restore
        self.start_epoch = 1
        checkpoint = None
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                self.model.load_state_dict(checkpoint, strict=True)

            self.start_epoch = 1 + checkpoint['epoch']
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            if self.trainer_params["load_optimizer"]:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(">> Optimizer (Epoch: {}) Loaded (lr = {})!".format(checkpoint['epoch'], self.optimizer.param_groups[0]['lr']))
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))
            print(">> Load from {}".format(checkpoint_fullname))

        # PID-Lagrangian lambda controller (dynamic penalty factor)
        if self.trainer_params.get("pid_lambda", False):
            self.pid_lambda_controller = PIDLambdaController(
                lambda_init=self.trainer_params.get("pid_lambda_init", 0.1),
                Kp=self.trainer_params.get("pid_lambda_kp", 0.1),
                Ki=self.trainer_params.get("pid_lambda_ki", 0.01),
                Kd=self.trainer_params.get("pid_lambda_kd", 0.0),
                target=self.trainer_params.get("pid_lambda_target", 0.0),
                ema_beta=self.trainer_params.get("pid_lambda_ema_beta", 0.9),
                lambda_min=0.0,
                lambda_max=self.trainer_params.get("pid_lambda_max", 10.0),
                integral_limit=None,
                signal_clip=(0.0, 1.0),
            )

            # Restore controller state from checkpoint if available
            if isinstance(checkpoint, dict) and "pid_lambda_state" in checkpoint:
                try:
                    self.pid_lambda_controller.load_state_dict(checkpoint["pid_lambda_state"])
                    print(">> PID lambda controller state loaded from checkpoint.")
                except Exception as e:
                    print(f">> Warning: failed to load PID lambda state from checkpoint: {e}")

            # PID lambda replaces penalty_factor
            self.penalty_factor = self.pid_lambda_controller.get_lambda()
            print(f">> PID lambda enabled. Initial lambda (penalty_factor) = {self.penalty_factor:.6f}")

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('================================================================================')


            # Update penalty factor if you want to train it in a curriculum learning way
            if self.trainer_params["penalty_increase"] and self.pid_lambda_controller is None:
                self.penalty_factor = 0.5 + epoch / self.trainer_params["epochs"] * 1.5

            # Train
            train_score, train_loss, infeasible = self._train_one_epoch(epoch)

            # Step
            self.scheduler.step()

            # Log
            if isinstance(train_score, list):
                dist_reward, total_timeout_reward, timeout_nodes_reward = train_score
                train_score = dist_reward
            if self.trainer_params["fsb_dist_only"]:
                try:
                    sol_infeasible_rate, ins_infeasible_rate, feasible_dist_mean, feasible_dist_max_pomo_mean = infeasible
                except:
                    pass
            else:
                sol_infeasible_rate, ins_infeasible_rate = infeasible

            # PID-lambda update (uses ins_infeasible_rate feedback)
            pid_info = None
            if self.pid_lambda_controller is not None and 'ins_infeasible_rate' in locals():
                update_interval = int(self.trainer_params.get("pid_lambda_update_interval", 1))
                if update_interval <= 0:
                    update_interval = 1
                if (epoch % update_interval) == 0:
                    try:
                        pid_info = self.pid_lambda_controller.step(float(ins_infeasible_rate))
                        self.penalty_factor = pid_info["lambda_val"]
                    except Exception as e:
                        print(f">> Warning: PID lambda update skipped due to error: {e}")
            if self.tb_logger:
                self.tb_logger.log_value('train/train_score', train_score, epoch)
                self.tb_logger.log_value('train/train_loss', train_loss, epoch)
                try:
                    self.tb_logger.log_value('feasibility/solution_infeasible_rate', sol_infeasible_rate, epoch)
                    self.tb_logger.log_value('feasibility/instance_infeasible_rate', ins_infeasible_rate, epoch)
                except:
                    pass
                if self.pid_lambda_controller is not None:
                    self.tb_logger.log_value('pid_lambda/lambda', self.penalty_factor, epoch)
                    if pid_info is not None:
                        self.tb_logger.log_value('pid_lambda/ema_error', pid_info["ema_error"], epoch)
                        self.tb_logger.log_value('pid_lambda/signal', pid_info["signal"], epoch)
                        self.tb_logger.log_value('pid_lambda/delta', pid_info["delta"], epoch)
                if self.trainer_params["timeout_reward"]:
                    self.tb_logger.log_value("feasibility/total_timeout", total_timeout_reward, epoch)
                    self.tb_logger.log_value("feasibility/timeout_nodes", timeout_nodes_reward, epoch)
                if self.trainer_params["fsb_dist_only"]:
                    self.tb_logger.log_value("feasibility/feasible_dist_mean", feasible_dist_mean, epoch)
                    self.tb_logger.log_value("feasibility/feasible_dist_max_pomo_mean", feasible_dist_max_pomo_mean, epoch)
            if self.wandb_logger:
                wandb.log({'train/train_score': train_score})
                wandb.log({'train/train_loss': train_loss})
                try:
                    wandb.log({'feasibility/solution_infeasible_rate': sol_infeasible_rate})
                    wandb.log({'feasibility/instance_infeasible_rate': ins_infeasible_rate})
                except:
                    pass
                if self.pid_lambda_controller is not None:
                    wandb.log({'pid_lambda/lambda': self.penalty_factor})
                    if pid_info is not None:
                        wandb.log({'pid_lambda/ema_error': pid_info["ema_error"]})
                        wandb.log({'pid_lambda/signal': pid_info["signal"]})
                        wandb.log({'pid_lambda/delta': pid_info["delta"]})
                if self.trainer_params["timeout_reward"]:
                    wandb.log({"feasibility/total_timeout": total_timeout_reward})
                    wandb.log({"feasibility/timeout_nodes": timeout_nodes_reward})
                if self.trainer_params["fsb_dist_only"]:
                    wandb.log({"feasibility/feasible_dist_mean": feasible_dist_mean})
                    wandb.log({"feasibility/feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean})

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            validation_interval = self.trainer_params['validation_interval']

            # Validation & save latest images
            try:
                if train_score < best_score:
                    best_score = train_score
                    torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_best.pt"))
                    print(">> Best model saved!")
            except:
                best_score = train_score

            # Save model
            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'problem': self.args.problem,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log,
                }
                if self.pid_lambda_controller is not None:
                    checkpoint_dict["pid_lambda_state"] = self.pid_lambda_controller.state_dict()
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))


            # validation
            if epoch == 1 or (epoch % validation_interval == 0):
                val_problems = [self.args.problem]
                val_episodes, problem_size = self.env_params['val_episodes'], self.env_params['problem_size']
                if self.env_params['val_dataset'] is not None:
                    paths = self.env_params['val_dataset']
                    dir = ["../data/{}/".format(self.args.problem)] * len(paths)
                    val_envs = [get_env(prob)[0] for prob in val_problems] * len(paths)
                else:
                    dir = [os.path.join("../data", prob) for prob in val_problems]
                    paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in val_problems]
                    val_envs = [get_env(prob)[0] for prob in val_problems]
                for i, path in enumerate(paths):
                    # if no optimal solution provided, set compute_gap to False
                    if not self.env_params["pomo_start"]:
                        # sampling pomo_size routes is useless due to the argmax operator when selecting next node based on probability
                        init_pomo_size = self.env_params["pomo_size"]
                        self.env_params["pomo_size"] = 1
                    score, gap, infsb_rate = self._val_and_stat(dir[i], path, val_envs[i](**self.env_params), batch_size=self.trainer_params["validation_batch_size"], val_episodes=val_episodes, epoch = epoch)
                    if not self.env_params["pomo_start"]:
                        self.env_params["pomo_size"] = init_pomo_size
                    if score is not None:  # Only append if validation was successful (dataset exists)
                        self.result_log["val_score"].append(score)
                        self.result_log["val_gap"].append(gap)
                        if infsb_rate is not None:
                            self.result_log["val_infsb_rate"].append(infsb_rate)
                        if self.tb_logger:
                            self.tb_logger.log_value('val_score/{}'.format(path.split(".")[0]), score, epoch)
                            self.tb_logger.log_value('val_gap/{}'.format(path.split(".")[0]), gap, epoch)
                            try:
                                self.tb_logger.log_value('val_sol_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[0], epoch)
                                self.tb_logger.log_value('val_ins_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[1], epoch)
                            except:
                                pass
                    if self.wandb_logger:
                        wandb.log({'val_score/{}'.format(path.split(".")[0]): score})
                        wandb.log({'val_gap/{}'.format(path.split(".")[0]): gap})
                        try:
                            wandb.log({'val_sol_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[0]})
                            wandb.log({'val_ins_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[1]})
                        except:
                            pass

                    try:
                        if score < best_val_score:
                            best_val_score = score
                            torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_val_best.pt"))
                            print(">> Best model on validation dataset saved!")
                    except:
                        best_val_score = score

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM, sol_infeasible_rate_AM, ins_infeasible_rate_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        if self.trainer_params["fsb_dist_only"]:
            feasible_dist_mean_AM, feasible_dist_max_pomo_mean_AM = AverageMeter(), AverageMeter()
        if self.trainer_params["timeout_reward"]:
            timeout_AM, timeout_nodes_AM = AverageMeter(), AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        total_step = math.floor(train_num_episode /self.trainer_params['train_batch_size'])
        batch_id = 0
        while episode < train_num_episode:
            for accumulation_step in range(self.trainer_params['accumulation_steps']):
                remaining = train_num_episode - episode
                batch_size = min(self.trainer_params['train_batch_size'], remaining)

                env = random.sample(self.envs, 1)[0](**self.env_params)
                data = env.get_random_problems(batch_size, self.env_params["problem_size"])

                avg_score, avg_loss, infeasible = self._train_one_batch(data, env, accumulation_step=accumulation_step)

                if isinstance(infeasible, dict):
                    sol_infeasible_rate = infeasible["sol_infeasible_rate"]
                    ins_infeasible_rate = infeasible["ins_infeasible_rate"]
                    try:
                        feasible_dist_mean, feasible_dist_mean_num = infeasible["feasible_dist_mean"]
                        feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num = infeasible["feasible_dist_max_pomo_mean"]
                        feasible_dist_mean_AM.update(feasible_dist_mean, feasible_dist_mean_num)
                        feasible_dist_max_pomo_mean_AM.update(feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num)
                    except:
                        pass
                else:
                    infeasible_rate = infeasible

                if isinstance(avg_score, list):
                    dist_reward, total_timeout_reward, timeout_nodes_reward = avg_score
                    avg_score = dist_reward
                    timeout_AM.update(total_timeout_reward, batch_size)
                    timeout_nodes_AM.update(timeout_nodes_reward, batch_size)
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                try:
                    sol_infeasible_rate_AM.update(sol_infeasible_rate, batch_size)
                    ins_infeasible_rate_AM.update(ins_infeasible_rate, batch_size)
                except:
                    pass

                episode += batch_size
                batch_id += 1
                if episode >= train_num_episode:
                    break

        # Log Once, for each epoch

        if False:  # PIP-D removed, always use else block
            if self.trainer_params["timeout_reward"]:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Timeout: {:.4f}, Timeout_nodes: {:.0f}, Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                               sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100, timeout_AM.avg,
                        timeout_nodes_AM.avg, feasible_dist_max_pomo_mean_AM.avg))
            else:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                        sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100,
                        feasible_dist_max_pomo_mean_AM.avg))
            print('Epoch {:3d}: PIP-D Loss: {:.4f},  Accuracy: {:.4f}% (BSF: {:.4f}%) [Infeasible: {:.4f}% (BSF: {:.4f}%), Feasible: {:.4f}% (BSF: {:.4f}%)]'.format(epoch, sl_loss_AM.avg, accuracy_AM.avg*100, self.accuracy_bsf*100, infsb_accuracy_AM.avg*100, self.infsb_accuracy_bsf*100, fsb_accuracy_AM.avg*100, self.fsb_accuracy_bsf*100))

            if self.tb_logger:
                self.tb_logger.log_value('sl_epoch/sl_loss', sl_loss_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/accuracy', accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/infsb_accuracy', infsb_accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/infsb_samples_number', infsb_accuracy_AM.count, epoch)
                self.tb_logger.log_value('sl_epoch/fsb_accuracy', fsb_accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/fsb_samples_number', fsb_accuracy_AM.count, epoch)
            if self.wandb_logger:
                wandb.log({'sl_epoch/sl_loss': sl_loss_AM.avg})
                wandb.log({'sl_epoch/accuracy': accuracy_AM.avg})
                wandb.log({'sl_epoch/infsb_accuracy': infsb_accuracy_AM.avg})
                wandb.log({'sl_epoch/infsb_samples_number': infsb_accuracy_AM.count})
                wandb.log({'sl_epoch/fsb_accuracy': fsb_accuracy_AM.avg})
                wandb.log({'sl_epoch/fsb_samples_number': fsb_accuracy_AM.count})

            # save lazy model every epoch
            if self.trainer_params["pip_save"] == "epoch":
                self.accuracy_isbsf = True if accuracy_AM.avg > self.accuracy_bsf else False
                self.fsb_accuracy_isbsf = True if fsb_accuracy_AM.avg > self.fsb_accuracy_bsf else False
                self.infsb_accuracy_isbsf = True if infsb_accuracy_AM.avg > self.infsb_accuracy_bsf else False

                self.accuracy_bsf = accuracy_AM.avg if accuracy_AM.avg > self.accuracy_bsf else self.accuracy_bsf
                self.fsb_accuracy_bsf = fsb_accuracy_AM.avg if fsb_accuracy_AM.avg > self.fsb_accuracy_bsf else self.fsb_accuracy_bsf
                self.infsb_accuracy_bsf = infsb_accuracy_AM.avg if infsb_accuracy_AM.avg > self.infsb_accuracy_bsf else self.infsb_accuracy_bsf

                if self.accuracy_isbsf:
                    if not os.path.exists('{}/accuracy_bsf.pt'.format(self.log_path)) or (infsb_accuracy > 0.75 and fsb_accuracy > 0.75):
                        # if not exist, save
                        # then check whether the current batch is bad, if no then save
                        print("Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(self.accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(self.log_path))
                if self.fsb_accuracy_isbsf:
                    if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(self.log_path)) or infsb_accuracy > 0.75 or  (infsb_accuracy > 0.6 and self.problem=="TSPDL"):
                        # if not exist, save
                        # then check whether the current batch is bad, if yes then don't save
                        print("Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format( self.fsb_accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(self.log_path))
                if self.infsb_accuracy_isbsf:
                    if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(self.log_path)) or fsb_accuracy > 0.75 or (fsb_accuracy > 0.6 and self.problem=="TSPDL"):
                        # if not exist, save
                        # then check whether the current batch is bad, if yes then don't save
                        print("Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(self.infsb_accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(self.log_path))
        else:
            if self.trainer_params["timeout_reward"]:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Timeout: {:.4f}, Timeout_nodes: {:.0f}, Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                        sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100, timeout_AM.avg,
                        timeout_nodes_AM.avg, feasible_dist_max_pomo_mean_AM.avg))
            else:
                try:
                    print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Feasible_dist: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, sol_infeasible_rate_AM.avg*100, ins_infeasible_rate_AM.avg*100, feasible_dist_max_pomo_mean_AM.avg))
                except:
                    print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        if self.trainer_params["fsb_dist_only"]:
            try:
                infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg, feasible_dist_mean_AM.avg, feasible_dist_max_pomo_mean_AM.avg]
            except:
                infeasible_output = None
        else:
            infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg]

        if self.trainer_params["timeout_reward"]:
            score_output = [score_AM.avg, timeout_AM.avg, timeout_nodes_AM.avg]
        else:
            score_output = score_AM.avg

        return score_output, loss_AM.avg, infeasible_output

    def _train_one_batch(self, data, env, accumulation_step):

        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])

        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)

        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))

        ########################################
        ############ POMO Rollout ##############
        ########################################
        state, reward, done = env.pre_step()
        while not done:
            # Forward
            selected, prob = self.model(state, pomo=self.env_params["pomo_start"],
                                            tw_end = env.node_tw_end if self.problem == "STSPTW" else None)

            # Step
            state, reward, done, infeasible = env.step(selected,
                                                       out_reward = self.trainer_params["timeout_reward"],
                                                       generate_PI_mask = self.model_params["generate_PI_mask"],
                                                       pip_step = self.trainer_params["pip_step"])
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))

            # Handle outputs
            if isinstance(infeasible, list):
                infeasible, infsb_level_value = infeasible
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)




        ########################################
        ############ Calculate Loss ############
        ########################################
        # Rewards calculation
        infeasible_output = infeasible
        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            dist = dist_reward.clone()
        else:
            dist_reward = reward
            dist = reward
        if self.trainer_params["fsb_dist_only"]:
            problem_size, pomo_size = self.env_params["problem_size"], env.pomo_size
            feasible_number = (batch_size*pomo_size) - infeasible.sum()
            feasible_dist_mean, feasible_dist_max_pomo_mean = 0., 0.
            batch_feasible = torch.tensor([0.])
            if feasible_number:
                feasible_dist = torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward) # feasible dist left only
                feasible_dist_mean = -feasible_dist.sum() / feasible_number # negative sign to make positive value, and calculate mean
                feasible_dist_mean = (feasible_dist_mean, feasible_number)
                reward_masked = dist.masked_fill(infeasible, -1e10)  # get feasible results from pomo
                feasible_max_pomo_dist = reward_masked.max(dim=1)[0]# get best results from pomo, shape: (batch)
                # feasible_max_pomo_dist = dist.max(dim=1)[0] # get best results from pomo, shape: (batch)
                batch_feasible = (infeasible==False).any(dim=-1) # shape: (batch)
                feasible_max_pomo_dist = torch.where(batch_feasible==False, torch.zeros_like(feasible_max_pomo_dist), feasible_max_pomo_dist) # feasible dist left only
                feasible_dist_max_pomo_mean = -feasible_max_pomo_dist.sum() / batch_feasible.sum() # negative sign to make positive value, and calculate mean
                feasible_dist_max_pomo_mean = (feasible_dist_max_pomo_mean, batch_feasible.sum())

            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (batch_size*pomo_size),
                "ins_infeasible_rate": 1. - batch_feasible.sum() / batch_size,
                "feasible_dist_mean": feasible_dist_mean,
                "feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean
            }
        if isinstance(reward, list):
            reward = dist +  self.penalty_factor * (total_timeout_reward +  timeout_nodes_reward)  # (batch, pomo)
        if not self.trainer_params["timeout_reward"] and self.trainer_params["fsb_reward_only"]: # activate when not using LM
            feasible_reward_number = (infeasible==False).sum(-1)
            feasible_reward_mean = (torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward).sum(-1) / feasible_reward_number)[:,None]
            feasible_advantage = dist_reward - feasible_reward_mean
            feasible_advantage = torch.masked_select(feasible_advantage, infeasible==False)
            log_prob = torch.masked_select(prob_list.log().sum(dim=2), infeasible==False)
            advantage = feasible_advantage
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()

        ########################################
        ############# Step & Return ############
        ########################################
        if accumulation_step == 0:
            self.model.zero_grad()
        loss_mean = loss_mean/self.trainer_params["accumulation_steps"]
        loss_mean.backward()
        if accumulation_step == self.trainer_params["accumulation_steps"] - 1:
            # update the parameters until accumulating enough accumulation_steps
            self.optimizer.step()

        if not self.trainer_params["timeout_reward"]:
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
            score_mean = score_mean.item()
        else:
            max_dist_reward = dist_reward.max(dim=1)[0]  # get best results from pomo
            dist_mean = -max_dist_reward.float().mean()  # negative sign to make positive value
            max_timeout_reward = total_timeout_reward.max(dim=1)[0]  # get best results from pomo
            timeout_mean = -max_timeout_reward.float().mean()  # negative sign to make positive value
            max_timeout_nodes_reward = timeout_nodes_reward.max(dim=1)[0]  # get best results from pomo
            timeout_nodes_mean = -max_timeout_nodes_reward.float().mean()  # negative sign to make positive value
            score_mean = [dist_mean, timeout_mean, timeout_nodes_mean]

        loss_out = loss_mean.item()
        return score_mean, loss_out, infeasible_output

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.model.eval()
        self.model.set_eval_type(eval_type)

        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor, normalize=True)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, prob = self.model(state, pomo=self.env_params["pomo_start"],
                                               tw_end = env.node_tw_end if self.problem == "STSPTW" else None)
                # shape: (batch, pomo)
                state, reward, done, infeasible = env.step(selected,
                                                           generate_PI_mask=self.trainer_params["generate_PI_mask"],
                                                           pip_step = self.trainer_params["pip_step"])

        # Return
        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            dist = dist_reward.clone()

            aug_total_timeout_reward = total_timeout_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_total_timeout_reward, _ = aug_total_timeout_reward.max(dim=2)  # get best results from pomo
            no_aug_total_timeout_score = -max_pomo_total_timeout_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_total_timeout_reward, _ = max_pomo_total_timeout_reward.max(dim=0)  # get best results from augmentation
            aug_total_timeout_score = -max_aug_pomo_total_timeout_reward.float()  # negative sign to make positive value

            aug_timeout_nodes_reward = timeout_nodes_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_timeout_nodes_reward, _ = aug_timeout_nodes_reward.max(dim=2)  # get best results from pomo
            no_aug_timeout_nodes_score = -max_pomo_timeout_nodes_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_timeout_nodes_reward, _ = max_pomo_timeout_nodes_reward.max(dim=0)  # get best results from augmentation
            aug_timeout_nodes_score = -max_aug_pomo_timeout_nodes_reward.float()  # negative sign to make positive value
        else:
            dist = reward

        aug_reward = dist.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)

        if self.trainer_params["fsb_dist_only"]:
            # shape: (augmentation, batch, pomo)
            infeasible = infeasible.reshape(aug_factor, batch_size, env.pomo_size)  # shape: (augmentation, batch, pomo)
            no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)  # shape: (batch)
            aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)  # shape: (batch)

            reward_masked = aug_reward.masked_fill(infeasible, -1e10) # get feasible results from pomo
            fsb_no_aug = reward_masked[0,:,:].max(dim=1, keepdim=True).values # shape: (augmentation, batch)
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug

            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (batch_size * env.pomo_size * aug_factor),
                "ins_infeasible_rate": 1. - aug_feasible.sum() / batch_size,
                "no_aug_feasible": no_aug_feasible,
                "aug_feasible": aug_feasible
            }
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
            infeasible_output = infeasible

        return no_aug_score, aug_score, infeasible_output

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False, epoch=1):
        # Check if validation dataset file exists
        dataset_path = os.path.join(dir, val_path)
        if not os.path.exists(dataset_path):
            print(f">> Warning: Validation dataset not found at {dataset_path}, skipping validation...")
            return None, None, None
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list, sol_infeasible_rate_list, ins_infeasible_rate_list = [], [], [], [], [], []
        episode, no_aug_score, aug_score, sol_infeasible_rate, ins_infeasible_rate = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        used_data = None
        # if self.trainer_params["timeout_reward"]:
        #     no_aug_total_timeout_score, no_aug_timeout_nodes_score = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        #     aug_total_timeout_score, aug_timeout_nodes_score = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.trainer_params["fsb_dist_only"]:
            no_aug_feasible, aug_feasible = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.trainer_params["use_real_PI_mask"] and self.model_params["generate_PI_mask"]:
            print(">> Use PI masking for validation...")

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)
            if data is None:
                # reached end of dataset
                break
            used_data = data
            no_aug, aug, infsb_rate = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            if isinstance(aug, list):
                no_aug, no_aug_total_timeout, no_aug_timeout_nodes = no_aug
                aug, aug_total_timeout, aug_timeout_nodes = aug
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            if isinstance(infsb_rate, dict):
                no_aug_fsb = infsb_rate["no_aug_feasible"]
                aug_fsb = infsb_rate["aug_feasible"]
                sol_infsb_rate = infsb_rate["sol_infeasible_rate"]
                ins_infsb_rate = infsb_rate["ins_infeasible_rate"]
                no_aug_feasible = torch.cat((no_aug_feasible, no_aug_fsb), dim=0)
                aug_feasible = torch.cat((aug_feasible, aug_fsb), dim=0)
            try:
                sol_infeasible_rate = torch.cat((sol_infeasible_rate, torch.tensor([sol_infsb_rate])), dim=0)
                ins_infeasible_rate = torch.cat((ins_infeasible_rate, torch.tensor([ins_infsb_rate])), dim=0)
            except:
                pass
            # advance by actual loaded batch size (dataset might be smaller than requested)
            episode += data[-1].size(0)
        if self.trainer_params["fsb_dist_only"]:
            print(">> Only feasible solutions are under consideration!")
            no_aug_score_list.append(round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4))
            aug_score_list.append(round(aug_score[aug_feasible.bool()].mean().item(), 4))
        else:
            no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
            aug_score_list.append(round(aug_score.mean().item(), 4))
        if sol_infeasible_rate.size(0) > 0:
            sol_infeasible_rate_list.append(round(sol_infeasible_rate.mean().item()*100, 3))
            ins_infeasible_rate_list.append(round(ins_infeasible_rate.mean().item() * 100, 3))

        try:
            problem_size = env.problem_size if hasattr(env, "problem_size") else (used_data[1].size(1) if used_data is not None else None)
            sol_path = get_opt_sol_path(dir, env.problem, problem_size)
        except:
            sol_path = os.path.join(dir, "lkh_" + val_path)

        compute_gap = os.path.exists(sol_path)

        if compute_gap:
            opt_sol = load_dataset(sol_path, disable_print=True)[: episode]
            # grid_factor = 1.
            grid_factor = 100. if self.args.problem == "STSPTW" else 1.
            opt_sol = torch.tensor([i[0]/grid_factor for i in opt_sol])
            if self.trainer_params["fsb_dist_only"]:
                gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
            else:
                gap = (no_aug_score - opt_sol) / opt_sol * 100
                aug_gap = (aug_score - opt_sol) / opt_sol * 100
            no_aug_gap_list.append(round(gap.mean().item(), 4))
            aug_gap_list.append(round(aug_gap.mean().item(), 4))
            try:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
                return aug_score_list[0], aug_gap_list[0], [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]
            except:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
                return aug_score_list[0], aug_gap_list[0], None

        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, aug_score_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
            return aug_score_list[0], 0, [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]

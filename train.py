import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from models.SCCNN import SCCNN
from univnet.generator import Generator
from univnet.discriminator import Discriminator
from Loss import MultiResolutionSTFTLoss, DurationLoss
import audio as Audio
from dataloader import prepare_dataloader
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils
import itertools


def load_checkpoint(checkpoint_path, model, voc_g, voc_d, g_optim, d_optim):
    assert os.path.isfile(checkpoint_path)
    print("Starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    if 'model' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model'])
        print('Model is loaded!')
    if 'voc_g' in checkpoint_dict:
        voc_g.load_state_dict(checkpoint_dict['voc_g'])
    if 'voc_d' in checkpoint_dict:
        voc_d.load_state_dict(checkpoint_dict['voc_d'], strict = False)
        print('Vocoder is loaded!')
    if 'g_optim' in checkpoint_dict:
        g_optim.load_state_dict(checkpoint_dict['g_optim'])
    if 'd_optim' in checkpoint_dict:
        d_optim.load_state_dict(checkpoint_dict['d_optim'])
        print('Optimizer is loaded!')
    current_step = checkpoint_dict['step'] + 1
    last_epoch = checkpoint_dict['epoch'] + 1
    
    return model, voc_g, voc_d, g_optim, d_optim, current_step, last_epoch


def main(args, c):

    # Used for calculating mel loss in evaluate.py
    STFT = Audio.stft.TacotronSTFT(c["filter_length"], c["hop_length"], c["win_length"],
                                   c["n_mel_channels"], c["sampling_rate"], c["mel_fmin"], c["mel_fmax"])
    
    # Define model
    model = SCCNN(c).cuda()
    voc_g = Generator(c).cuda()
    voc_d = Discriminator(c).cuda()
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model) + utils.get_param_num(voc_g) #* Prior 더하고 Posterior 제외해야 됨
    print('Number of Model Parameters:', num_param)
    
    with open(os.path.join(args.save_path, "model.txt"), "w") as f_log:
        f_log.write(str(model))

    g_optim = torch.optim.AdamW(itertools.chain(model.parameters(), voc_g.parameters()), c.learning_rate, betas=c.betas, eps=c.eps)
    d_optim = torch.optim.AdamW(voc_d.parameters(), c.learning_rate, betas=c.betas, eps=c.eps) #* JETS 혹은 다른 Single-Stage 참고
    
    # Define loss terms
    DurLoss = DurationLoss()
    resolutions = eval(c.mrd_resolutions_16k)
    STFTLoss = MultiResolutionSTFTLoss('cuda', resolutions)
    #* Prior 쪽 Loss
    print("Optimizer and Loss Function Defined.")

    # Get dataset
    data_loader = prepare_dataloader(args.data_path, "train.txt", shuffle=True, \
                                     batch_size=c.batch_size, segment_size=c.segment_size, split=True) 
    print("Data Loader is Prepared.")

    # Load checkpoint if exists
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        model, voc_g, voc_d, g_optim, d_optim, current_step, last_epoch= \
            load_checkpoint(args.checkpoint_path, model, voc_g, voc_d, g_optim, d_optim)
        print("\n---Model Restored at Step {}---\n".format(current_step))
    else:
        print("\n---Start New Training---\n")
        current_step = 0
        last_epoch = -1
    checkpoint_path = os.path.join(args.save_path, 'ckpt')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Scheduled optimizer
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(g_optim, gamma=c.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(d_optim, gamma=c.lr_decay, last_epoch=last_epoch)

    # Init logger
    log_path = os.path.join(args.save_path, 'log')
    logger = SummaryWriter(os.path.join(log_path, 'board'))
    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
        f_log.write("Dataset :{}\n Number of Parameters: {}\n".format(c.dataset, num_param))

    # Init synthesis directory
    synth_path = os.path.join(args.save_path, 'synth')
    os.makedirs(synth_path, exist_ok=True)

    # Training
    model.train()
    voc_g.train()
    voc_d.train()
    
    while current_step < args.max_iter:        
        # Get Training Loader
        for idx, batch in enumerate(data_loader):

            if current_step == args.max_iter:
                break
                
            #* Get Data
            sid, text, mel_target, latent, audio, indices, D, log_D, f0, energy, \
                    src_len, mel_len, latent_len, \
                        max_src_len, max_mel_len, max_latent_len = model.parse_batch(batch)

            # chunked GT waveforms
            y = torch.stack(audio) #* (b_s, segment length)
            y = torch.autograd.Variable(y.to('cuda', non_blocking=True)) # wav to device
            y = y.unsqueeze(1)
            indices = torch.tensor(indices).cuda()


            #* Forward
            # Encoder including Aligner
            i_output, src_output, speaker_vector, style_target, log_duration_output, align_maps, src_mask, mel_mask, _  = model(
                    text, src_len, mel_target, latent, f0, energy, mel_len, latent_len, \
                        D, max_src_len, max_mel_len, max_latent_len)

            # Vocoder
            x_hat = []
            for i, idx in enumerate(indices):
                chunk = i_output[i, idx:idx+y.size(-1)//c.hop_length, :] # intermediate feature
                x_hat.append(chunk)
            
            x_hat = torch.stack(x_hat).transpose(1,2).cuda()
            
            noise = torch.randn(x_hat.size(0), c.voc_noise_dim, x_hat.size(2)).cuda()
            y_g_hat = voc_g(x_hat, noise)
            
            # Prior
            
            
            #* Loss calculation and backpropagation
            # Discriminator_Vocoder - LSGAN loss
            voc_d_loss = 0.0
            
            if current_step <= c.mtrstft_only:
                pass
            
            else:
                d_optim.zero_grad()
                res_fake, period_fake = voc_d(y_g_hat.detach())
                res_real, period_real = voc_d(y)

                for (_, score_fake), (_, score_real) in zip(res_fake+period_fake, res_real+period_real):
                    voc_d_loss = voc_d_loss + torch.mean(torch.pow(score_real-1.0,2))
                    voc_d_loss = voc_d_loss + torch.mean(torch.pow(score_fake, 2))
                    
                voc_d_loss = voc_d_loss / len(res_fake+period_fake)
                
                voc_d_loss.backward()
                nn.utils.clip_grad_norm_(voc_d.parameters(), c.grad_clip_thresh)
                d_optim.step()
            
            # Generator_Vocoder + Aligner
            g_optim.zero_grad()

            # Duration loss
            d_loss = DurLoss(log_duration_output, log_D, src_len)

            # Vocoder reconstruction loss
            sc_loss, mag_loss = STFTLoss(y_g_hat.squeeze(1), y.squeeze(1))
            stft_loss = (sc_loss+mag_loss)*2.5
        
            # Prior loss

            
            total_loss = d_loss + stft_loss
            
            # Generator_Vocoder - LSGAN loss
            voc_g_loss = 0.0
            
            if current_step > c.mtrstft_only:
                
                res_fake, period_fake = voc_d(y_g_hat)
                voc_g_loss = 0.0
                for (_, score_fake) in res_fake+period_fake:
                    voc_g_loss += torch.mean(torch.pow(score_fake-1.0,2))
                voc_g_loss = voc_g_loss / len(res_fake+period_fake)
                
                total_loss = total_loss + voc_g_loss
                
            # Backward
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip_thresh)
            nn.utils.clip_grad_norm_(voc_g.parameters(), c.grad_clip_thresh)
            g_optim.step()

            # Print log
            if current_step % args.std_step == 0 and current_step != 0:    
                t_l = total_loss.item()
                d_l = d_loss.item()
                s_l = stft_loss.item()

                str1 = "Step [{}/{}], Epoch {}:".format(current_step, args.max_iter, last_epoch+1)
                str2 = "Total Loss: {:.4f}\tDuration Loss: {:.4f}\tMTRSTFT Loss: {:.4f};" \
                        .format(t_l, d_l, s_l)
                print(str1 + "\n" + str2 +"\n")
                
                if current_step % args.log_step == 0:  
                
                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n" + str2 +"\n")

                    logger.add_scalar('Train/total_loss', t_l, current_step)
                    logger.add_scalar('Train/duration_loss', d_l, current_step)
                    logger.add_scalar('Train/mtrstft_loss', s_l, current_step)
                    utils.plot_attn(logger,'Train/align_maps', align_maps, current_step, c.cross_attn_head)
            
            # Save Checkpoint
            if current_step % args.save_step == 0 and current_step != 0:
                torch.save({'model': model.state_dict(), 'voc_g': voc_g.state_dict(), 'voc_d': voc_d.state_dict(), \
                    'g_optim': g_optim.state_dict(), 'd_optim': d_optim.state_dict(), 'step': current_step, 'epoch': last_epoch}, 
                    os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                print("*** Save Checkpoint ***")
                print("Save model at step {}...\n".format(current_step))

            #! TODO: audio 생성 및 ... mel plot? ... 고민
            '''
            if current_step % args.eval_step == 0 and current_step != 0:
                model.eval()
                with torch.no_grad():
                    m_l, d_l, f_l, e_l = evaluate(args, model, current_step)
                    str_v = "*** Validation ***\n" \
                            "SCCNN Step {},\n" \
                            "Mel Loss: {}\nDuration Loss:{}\nF0 Loss: {}\nEnergy Loss: {}" \
                            .format(current_step, m_l, d_l, f_l, e_l)
                    print(str_v + "\n" )
                    with open(os.path.join(log_path, "eval.txt"), "a") as f_log:
                        f_log.write(str_v + "\n")
                    logger.add_scalar('Validation/mel_loss', m_l, current_step)
                    logger.add_scalar('Validation/duration_loss', d_l, current_step)
                    logger.add_scalar('Validation/f0_loss', f_l, current_step)
                    logger.add_scalar('Validation/energy_loss', e_l, current_step)
                model.train()
            '''

            if current_step % args.eval_step == 0 and current_step != 0:
             
                with torch.no_grad():
                    evaluate(args, c, model, voc_g, STFT, logger, current_step)

            
            current_step += 1 
            
        scheduler_g.step()
        scheduler_d.step()
        last_epoch += 1

    print("Training Done at Step : {}".format(current_step))
    torch.save({'model': model.state_dict(), 'voc_g': voc_g.state_dict(), 'voc_d': voc_d.state_dict(), \
        'g_optim': g_optim.state_dict(), 'd_optim': d_optim.state_dict(), 'step': current_step, 'epoch': last_epoch},
        os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/hcy71/DATA/preprocessed_data/LibriTTS_16k')
    parser.add_argument('--save_path', default='exp')
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--max_iter', default=1000000, type=int)
    parser.add_argument('--save_step', default=20000, type=int)
    parser.add_argument('--synth_step', default=20000, type=int)
    parser.add_argument('--eval_step', default=10000, type=int)
        # parser.add_argument('--eval_step', default=10000, type=int)
    # parser.add_argument('--test_step', default=10000, type=int)
    parser.add_argument('--log_step', default=1000, type=int)
    parser.add_argument('--std_step', default=50, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the pretrained model') 

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)
    utils.build_env(args.config, 'config.json', args.save_path)

    main(args, config)

import torch
from dataloader import prepare_dataloader
import audio as Audio
import torch.nn.functional as F
import os
from Loss import MultiResolutionSTFTLoss, DurationLoss


def evaluate(args, c, model, vocoder, STFT, logger, current_step):    
    # Get dataset
    data_loader = prepare_dataloader(args.data_path, "val.txt", batch_size=1, shuffle=False, split=False) 
    
    model.eval()
    vocoder.eval()
    torch.backends.cudnn.benchmark = False
    # Get loss function
    DurLoss = DurationLoss()
    resolutions = eval(c.mrd_resolutions_16k)
    STFTLoss = MultiResolutionSTFTLoss('cuda', resolutions)
    
    # Evaluation
    m_l_list = []
    s_l_list = []
    d_l_list = []
        
    for i, batch in enumerate(data_loader):
        # Get Data
        print(i)
        # if i > 100: break
        
        id_ = batch["id"]
        sid, text, mel_target, latent, audio, indices, D, log_D, f0, energy, \
                src_len, mel_len, latent_len, \
                    max_src_len, max_mel_len, max_latent_len = model.parse_batch(batch)

        audio = torch.stack(audio) #* (1, segment length)
        audio = torch.autograd.Variable(audio.to('cuda', non_blocking=True)) # wav to device
        audio = audio.unsqueeze(1)
    
        with torch.no_grad():
            # Forward
            i_output, src_output, speaker_vector, style_target, log_duration_output, src_mask, mel_mask, _  = model(
                    text, src_len, mel_target, latent, mel_len, latent_len, \
                        D, max_src_len, max_mel_len, max_latent_len)
            noise = torch.randn(1, c.voc_noise_dim, i_output.size(1)).cuda()
            fake_audio = vocoder(i_output.transpose(1,2),noise)[:, :, :audio.size(2)]
            print(fake_audio.shape, audio.shape)
            # mel_fake, _ = Audio.tools.get_mel_from_wav(fake_audio[0][0], STFT)
            mel_fake, _ = STFT.mel_spectrogram(fake_audio[0])
            mel_target, _ = STFT.mel_spectrogram(audio[0])
            print(mel_fake.shape, mel_target.shape)
            mel_loss = F.l1_loss(mel_fake.cuda(), mel_target.cuda())
            
            # Logger
            d_loss = DurLoss(log_duration_output, log_D, src_len)
            sc_loss, mag_loss = STFTLoss(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss+mag_loss)*2.5
            
            m_l = mel_loss.item()
            d_l = d_loss.item()
            s_l = stft_loss.item()
            
 
            m_l_list.append(m_l)
            d_l_list.append(d_l)
            s_l_list.append(s_l)

            if i < 10:
                fake_audio = fake_audio[0][0].cpu().detach().numpy()
                logger.add_audio('generated/y_{}'.format(i), fake_audio, current_step, c.sampling_rate)
                
                if current_step == args.eval_step: 
                    audio = audio[0][0].cpu().detach().numpy()
                    logger.add_audio('gt/y_{}'.format(i), audio, current_step, c.sampling_rate)
                    
    avg_m_l = sum(m_l_list) / len(m_l_list)
    avg_s_l = sum(s_l_list) / len(s_l_list)
    avg_d_l = sum(d_l_list) / len(d_l_list)
    
    str_v = "*** Validation ***\n" \
            "Step {}, Mel Loss:{}\n" \
            "MTRSTFT Loss:{}\nDuration Loss:{}" \
            .format(current_step, avg_m_l, avg_s_l, avg_d_l)
    print(str_v + "\n" )
    
    log_path = os.path.join(args.save_path, 'log')
    with open(os.path.join(log_path, "eval.txt"), "a") as f_log:
        f_log.write(str_v + "\n")
        
    logger.add_scalar('Validation/mel_loss', avg_m_l, current_step)
    logger.add_scalar('Validation/mtrstft_loss', avg_s_l, current_step)
    logger.add_scalar('Validation/duration_loss',avg_d_l, current_step)
    
    model.train()
    vocoder.train()
    torch.backends.cudnn.benchmark = True
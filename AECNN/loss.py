from util.utils import frame_to_audio, audio_to_frame, DFT_matrix
import torch
from torch.autograd import Variable


def freqLoss_From_TimeDomain(enhanced_out, clean_spec, n_fft=512, need_mean=True, device='cuda'):
    '''

    :param enhanced_out: (2048, frame_num)
    :param clean_spec: complex
    :param n_fft:
    :return: loss
    '''
    assert 'Complex' in torch.typename(clean_spec)
    D = DFT_matrix(n_fft)
    D_tensor = torch.from_numpy(D).requires_grad_()
    D_tensor = D_tensor.to(device)
    Dr = torch.real(D_tensor).float()
    Di = torch.imag(D_tensor).float()
    enhanced_audio = frame_to_audio(enhanced_out,
                           frame_size=enhanced_out.shape[0],
                           window=torch.ones(2048),
                           frame_num=enhanced_out.shape[1],
                           hop_len=256,
                           device=device)

    frames = audio_to_frame(audio=enhanced_audio,
                            frame_size=512,
                            window=torch.hamming_window(512),
                            hop_len=int(512/2),
                            device=device)
    pred_real = torch.matmul(Dr, frames)
    pred_image = torch.matmul(Di, frames)

    clean_real = torch.real(clean_spec)
    clean_image = torch.imag(clean_spec)
    loss = torch.abs(torch.abs(pred_real)+torch.abs(pred_image) - torch.abs(clean_real) - torch.abs(clean_image))
    loss = torch.sum(loss, dim=0) / loss.shape[0]
    if need_mean:
        loss = torch.mean(loss)
    else:
        loss = torch.sum(loss)
    return loss

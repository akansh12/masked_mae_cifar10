from einops import rearrange

def mae_loss_function(input_img, pred_img, mask, patch_size=2):
    """
    input_img: [B, C, H, W]
    pred_img: [B, npatch, patch_size*patch_size*C]
    mask: [B, npatch]    
    """
    #reshape input_img
    input_img = rearrange(input_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

    loss = (input_img - pred_img) ** 2
    print(loss.shape)
    loss = loss.mean(dim=-1)  
    print(loss.shape)

    loss = (loss * mask).sum() / mask.sum()
    return loss
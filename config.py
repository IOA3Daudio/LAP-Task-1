
class DefaultConfig(object):
    """
	This is a class for configurations
	"""

    # GPU list
    device_ids = [0]
    # Batch size
    batch_size = 80
    # EPOCHS
    epochs = 1024
    # Original Dataset Path
    dataset_mat_pth = './dataset mats/train_data.mat'
    fb_mat_pth = './dataset mats/target_dtf.mat'
    e_mat_pth = './dataset mats/target_e.mat'
    ft_mat_pth = './dataset mats/ft.mat'
    # Output Save Path
    model_save_pth = './model/'
    hrirs_out_pth = './pred_hrir_pp/'
    loss_png_pth = './loss/'

    # weight of the classification loss function
    cls_loss_w = 1 
    
    # random seed
    seed = 660

    # evaluation position
    pos_sum = 126

    # scheduler_step_size = 200
    # scheduler_gamma = 0.9
    

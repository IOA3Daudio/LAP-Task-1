import matplotlib.pyplot as plt


def plot_loss(config,history,pp):
    plt.clf()
    plt.plot(history['Train Loss'][5:],label = 'Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(config.loss_png_pth +'Loss_'+str(pp)+'.png')
    print(min(history['Train Loss']))
    best_model_ind = history['Train Loss'].index(min(history['Train Loss']))
    #.index(min(history['Valid Loss'][-int(EPOCHS/4):]))
    print(best_model_ind)
    print(history['Train Loss'][best_model_ind])
    return best_model_ind
import matplotlib.pyplot as plt


def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    print(img.shape)
    img = img.permute(1,2,0)
    print(img.shape)
    plt.imshow(img) #channel are at the last in matplotlib where it was at front in tensors
    #plt.show()

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        #plt.show()
        break
import torch
import matplotlib.pyplot as plt

classes = ["pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea","english marigold","tiger lily","moon orchid","bird of paradise","monkshood","globe thistle","snapdragon","colt's foot","king protea","spear thistle","yellow iris","globe-flower","purple coneflower","peruvian lily","balloon flower","giant white arum lily","fire lily","pincushion flower","fritillary","red ginger","grape hyacinth","corn poppy","prince of wales feathers","stemless gentian","artichoke","sweet william","carnation","garden phlox","love in the mist","mexican aster","alpine sea holly","ruby-lipped cattleya","cape flower","great masterwort","siam tulip","lenten rose","barbeton daisy","daffodil","sword lily","poinsettia","bolero deep blue","wallflower","marigold","buttercup","daisy","common dandelion","petunia","wild pansy","primula","sunflower","pelargonium","bishop of llandaff","gaura","geranium","orange dahlia","pink-yellow dahlia","cautleya spicata","japanese anemone","black-eyed susan","silverbush","californian poppy","osteospermum","spring crocus","bearded iris","windflower","tree poppy","gazania","azalea","water lily","rose","thorn apple","morning glory","passion flower","lotus","toad lily","anthurium","frangipani","clematis","hibiscus","columbine","desert-rose","tree mallow","magnolia","cyclamen","watercress","canna lily","hippeastrum","bee balm","ball moss","foxglove","bougainvillea","camellia","mallow","mexican petunia","bromelia","blanket flower","trumpet creeper","blackberry lily"]

def acc_loss(train_obj, test_obj):
    """Display the Train Loss and Accuracy graph. Test Loss and Accuracy graph."""
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot([x.item() for x in train_obj.train_loss])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_obj.train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_obj.test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_obj.test_acc)
    axs[1, 1].set_title("Test Accuracy")

def testvtrain(train_obj, test_obj):
    """Display Test vs Train Accuracy plot"""
    plt.axes(xlabel= 'epochs', ylabel= 'Accuracy')
    plt.plot(train_obj.train_endacc)
    plt.plot(test_obj.test_acc)
    plt.title('Test vs Train Accuracy')
    plt.legend(['Train', 'Test'])

def class_acc(model,device, test_loader):            
    class_correct = list(0. for i in range(102))
    class_total = list(0. for i in range(102))

    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(102):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
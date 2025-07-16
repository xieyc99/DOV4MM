from utils.util import *
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import random
from loaders.diffaugment import *
import encoder_model.vit4mae as vit4mae
import torch.optim as optim
from models_decoder import MaskedDecoderViT, MaskedAutoencoderViT2

# train param
dataset = 'ImageNet'  # defender's public dataset D_{pub}
method = 'MAE'  # the self-supervised method used by suspicious model M_{sus}
model_dataset = 'ImageNet'  # the pre-trained dataset used by suspicious model M_{sus}
model = 'vit_base'  # the architecture of suspicious model M_{sus}
n_sample_train_1 = 20000
n_sample_train_2 = 1000
model_path = rf'E:\ckpt\DOV4MIM\decoder\{method}\{model}\{dataset}_{model_dataset}_{n_sample_train_1}_{n_sample_train_2}\model_feature_50.pth'  # the save path of M_r
seed = 99
align_feature = True

# test param
sample_num = 30  # the iterations of sampling (K)
sample_size = 1024  # the sample number per iteration (N)
batch_size = 256  # the batch size for inference

set_seed(seed)
print('model_path:', model_path)

img_size = 192 if 'swin' in model else 224

if dataset == 'ImageNet':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=True)
    # train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=False)
elif dataset == 'ImageNet-50':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=True, sub_class_num=50)
    # train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=False, sub_class_num=50, selected_classes=train_dataset.selected_classes)
elif dataset == 'ImageNet-100':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=True, sub_class_num=100)
    # train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = ImageNet(root='D:\Exp\datasets\imagenet', transform=normal_transform, train=False, sub_class_num=100, selected_classes=train_dataset.selected_classes)
elif dataset == 'Food101':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = datasets.Food101(root='D:\Exp\datasets', split='train', download=True, transform=normal_transform)
    test_dataset = datasets.Food101(root='D:\Exp\datasets', split='test', download=True, transform=normal_transform)
elif dataset == 'COCO2017':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = COCO2017(r'D:\Exp\datasets\COCO2017\train2017\train2017', transform=normal_transform)
    test_dataset = COCO2017(r'D:\Exp\datasets\COCO2017\test2017\test2017', transform=normal_transform)
elif dataset == 'Place365':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean,std)

    normal_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    train_dataset = datasets.ImageFolder(r'D:\Exp\datasets\places365_standard\train', transform=normal_transform)
    test_dataset = datasets.ImageFolder(r'D:\Exp\datasets\places365_standard\val', transform=normal_transform)

if __name__=='__main__':
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))

    indices = np.random.permutation(len(train_dataset))
    indices_train_dataset_1 = indices[:n_sample_train_1]
    print('indices_train_dataset_1:', indices_train_dataset_1[:10])
    indices_train_dataset_2 = indices[n_sample_train_1:]
    train_dataset_1 = Subset(train_dataset, indices_train_dataset_1)
    train_dataset_2 = Subset(train_dataset, indices_train_dataset_2)

    print('train_dataset_1:', len(train_dataset_1))
    print('train_dataset_2:', len(train_dataset_2))
    
    # model = MaskedAutoencoderViT2(need_encoder=False, encoder_model=model, align_feature=align_feature).cuda()
    if 'swin' in model or 'mixmim' in model:
        model = MaskedAutoencoderViT2(img_size=img_size, patch_size=32, need_encoder=False, encoder_model=model, encoder_method=method, align_feature=align_feature).cuda()
    else:
        model = MaskedAutoencoderViT2(need_encoder=False, encoder_model=model, encoder_method=method, align_feature=align_feature).cuda()

    checkpoint = torch.load(model_path, map_location="cuda")
    msg = model.load_state_dict(checkpoint, strict=False)
    print('msg:', msg)

    model.eval()
    model.cuda()

    pbar = tqdm(total=sample_num, desc='Processing')

    train1_loss = AverageMeter()
    train2_loss = AverageMeter()
    test_loss = AverageMeter()
    delta_train2_train1_loss = []
    delta_test_train1_loss = []

    set_seed(0)

    with torch.no_grad():
        for i in range(sample_num):
            indices = random.sample(range(len(train_dataset_1)), sample_size)
            subset_train_1 = Subset(train_dataset_1, indices)
            data_loader_train_1 = DataLoader(subset_train_1, batch_size=batch_size, shuffle=False)

            indices = random.sample(range(len(train_dataset_2)), sample_size)
            subset_train_2 = Subset(train_dataset_2, indices)
            data_loader_train_2 = DataLoader(subset_train_2, batch_size=batch_size, shuffle=False)

            indices = random.sample(range(len(test_dataset)), sample_size)
            subset_test = Subset(test_dataset, indices)
            data_loader_test = DataLoader(subset_test, batch_size=batch_size, shuffle=False)

            train1_loss.reset()
            train2_loss.reset()
            test_loss.reset()

            for data in data_loader_train_1:
                img, _ = data
                img = img.cuda()

                _, loss_train_1 = model(img)

                train1_loss.update(loss_train_1.item())

            for data in data_loader_train_2:
                img, _ = data
                img = img.cuda()

                _, loss_train_2 = model(img)

                train2_loss.update(loss_train_2.item())

            for data in data_loader_test:
                img, _ = data
                img = img.cuda()

                _, loss_test = model(img)

                test_loss.update(loss_test.item())

            delta_1 = train2_loss.avg - train1_loss.avg
            delta_2 = test_loss.avg - train1_loss.avg
            
            delta_train2_train1_loss.append(delta_1)
            delta_test_train1_loss.append(delta_2)

            pbar.update()
        pbar.close()

    print('delta_train2_train1_loss:', np.max(delta_train2_train1_loss), np.min(delta_train2_train1_loss), np.mean(delta_train2_train1_loss))
    print('delta_test_train1_loss:', np.max(delta_test_train1_loss), np.min(delta_test_train1_loss), np.mean(delta_test_train1_loss))
    t, p = one_tailed_ttest(delta_test_train1_loss, delta_train2_train1_loss)
    print('t:', t, 'p:', p)


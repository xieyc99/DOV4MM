from utils.util import *
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import random
from loaders.diffaugment import *
import encoder_model.vit4mae as vit4mae
import torch.optim as optim
from models_decoder import MaskedDecoderViT, MaskedAutoencoderViT2

# args
ImageNet_ckpt_name = {'MAE':{'vit_base':r"mae_pretrain_vit_base.pth", 'vit_large':r"mae_pretrain_vit_large.pth"},
                      'CAE':{'cae_base':r"cae_base_1600ep.pth", 'cae_large':r"cae_large_1600ep.pth"},
                      'IBOT':{'vit_base':r"checkpoint_teacher.pth", 'vit_large':r"checkpoint_teacher.pth"},
                      'SimMIM':{'swin_base':r"simmim_pretrain__swin_base__img192_window6__800ep.pth", 'swin_large':r"simmim_pretrain__swin_large__img192_window12__800ep.pth"},
                      'MaskFeat':{'vit_base':r"maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth"},
                      'PixMIM':{'vit_base':r"pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230322-e8137924.pth"},
                      'BEIT':{'vit_base':r"beit_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221128-ab79e626.pth"},
                      'BEITv2':{'vit_base':r"beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221212-a157be30.pth"},
                      'MixMIM':{'mixmim_base':r"mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.pth"},
                      'EVA':{'vit_base':r"eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth"}}

dataset = 'ImageNet'  # defender's public dataset D_{pub}
method = 'MAE'  # the self-supervised method used by suspicious model M_{sus}
model_dataset = 'ImageNet'  # the pre-trained dataset used by suspicious model M_{sus}
model = 'vit_base'  # the architecture of suspicious model M_{sus}
encoder_path = rf"E:\ckpt\DOV4MIM\{method}\ImageNet\{model}\{ImageNet_ckpt_name[method][model]}"  # the weight path of suspicious model M_{sus}
batch_size = 64  # the batch size for training the reconstructor M_r
n_sample_train_1 = 20000  # the amount of training data for the reconstructor M_r
n_sample_train_2 = 1000
n_sample_test = 1000
epochs = 50  # the training epochs for M_r
lr = 1e-3  # the learning rate for training M_r
save_path = rf'E:\ckpt\DOV4MIM\decoder\{method}\{model}\{dataset}_{model_dataset}_{n_sample_train_1}_{n_sample_train_2}'  # the save path of M_r
seed = 99
align_feature = True
save_epoch = 50

set_seed(seed)
os.makedirs(save_path, exist_ok=True)
print('save_path:', save_path)

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
    indices_train_dataset_2 = indices[n_sample_train_1:n_sample_train_1+n_sample_train_2]
    train_dataset_1 = Subset(train_dataset, indices_train_dataset_1)
    train_dataset_2 = Subset(train_dataset, indices_train_dataset_2)
    train_loader_1 = DataLoader(Subset(train_dataset, indices_train_dataset_1), batch_size=batch_size, shuffle=True)
    train_loader_2 = DataLoader(Subset(train_dataset, indices_train_dataset_2), batch_size=batch_size, shuffle=True)

    indices = np.random.permutation(len(test_dataset))
    indices_test_dataset = indices[:n_sample_test]
    test_loader = DataLoader(Subset(test_dataset, indices_test_dataset), batch_size=batch_size, shuffle=True)

    print('train_dataset_1:', len(train_dataset_1))
    print('train_dataset_2:', len(train_dataset_2))
    
    if 'swin' in model or 'mixmim' in model:
        model = MaskedAutoencoderViT2(img_size=img_size, patch_size=32, encoder_model=model, encoder_path=encoder_path, encoder_method=method, align_feature=align_feature).cuda()
    else:
        model = MaskedAutoencoderViT2(encoder_model=model, encoder_path=encoder_path, encoder_method=method, align_feature=align_feature).cuda()
        

    train1_loss = AverageMeter()
    train2_loss = AverageMeter()
    test_loss = AverageMeter()
    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train1_loss.reset()
        start = time.time()

        with tqdm(train_loader_1, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as tepoch:
            for data, _ in tepoch:
                data = data.cuda()

                optimizer.zero_grad()

                output, loss = model(data)

                loss.backward()
                optimizer.step()

                train1_loss.update(loss.item())

                tepoch.set_postfix(train1_loss=round(loss.item(), 3))
        
        lr_scheduler.step()

        print('[{}-epoch] time:{:.3f} | train1 loss: {} |'.format(epoch + 1, time.time() - start, train1_loss.avg))

        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                train2_loss.reset()
                test_loss.reset()
                with tqdm(train_loader_2, desc=f'Train2 Testing', unit='batch') as tepoch:
                    start = time.time()
                    for data, _ in tepoch:
                        data = data.cuda()

                        output, loss = model(data)

                        train2_loss.update(loss.item())

                        tepoch.set_postfix(train2_loss=round(loss.item(), 3))

                    print('[{}-epoch] time:{:.3f} | train2 loss: {} |'.format(epoch + 1, time.time() - start, train2_loss.avg))
                
                with tqdm(test_loader, desc=f'Testing', unit='batch') as tepoch:
                    start = time.time()
                    for data, _ in tepoch:
                        data = data.cuda()

                        output, loss = model(data)

                        test_loss.update(loss.item())

                        tepoch.set_postfix(test_loss=round(loss.item(), 3))

                    print('[{}-epoch] time:{:.3f} | test loss: {} |'.format(epoch + 1, time.time() - start, test_loss.avg))

        if (epoch+1) % save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_feature_{epoch+1}.pth' if align_feature else f'model_{epoch+1}.pth'))
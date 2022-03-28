
# train dcgan for synthetic data
CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/dcgan.py --dataroot ./data/ --dataset synthetic --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/synthetic_grey/dcgan/ --manualSeed 108 --niter 200 --batchSize 64 --network alexnet  --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --num_classes 10 --synthetic_dir ./synthetic_dataset/50k_samples/ --total_synth_samples 50000 --grey_scale

mkdir val_data

CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --dcgan_netG ./cifar100_run_models/alexnet/synthetic_grey/dcgan/netG_epoch_199.pth --dcgan_out ./val_data/dcgan_val_data_synthetic_alexnet_grey.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth  --network alexnet

# train student
CUDA_VISIBLE_DEVICES=0 python ./code/train_student/train_student.py --from_scratch --dcgan_netG ./cifar100_run_models/alexnet/synthetic_grey/dcgan/netG_epoch_199.pth --count 1 --dcgan_data --dcgan_data_ratio 0.5  --proxy_data --proxy_data_ratio 0.5 --pad_crop --name proxy_dcgan_45k_synthetic_grey_imgs --val_data_dcgan ./val_data/dcgan_val_data_synthetic_alexnet_grey.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --network alexnet --proxy_dataset synthetic --true_dataset cifar10 --num_classes 10 --synthetic_dir ./synthetic_dataset/50k_samples/ --total_synth_samples 50000 --grey_scale 

# ./train_student/checkpoint/alexnet/cifar10_synthetic/1/last_epoch_best_student_model_proxy_dcgan_45k_synthetic_grey_imgs.pth

# Run div_gan

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_gen.py --dataroot ./data/  --dataset synthetic --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/out_step2_0_10/synthetic_grey/2/ --manualSeed 108 --niter 100 --batchSize 64 --netG ./cifar100_run_models/alexnet/synthetic_grey/dcgan/netG_epoch_199.pth --student_path ./train_student/checkpoint/alexnet/cifar10_synthetic/1/last_epoch_best_student_model_proxy_dcgan_45k_synthetic_grey_imgs.pth  --network alexnet --c_l 0 --d_l 10 --true_dataset cifar10 --num_classes 10 --synthetic_dir ./synthetic_dataset/50k_samples/ --total_synth_samples 50000 --grey_scale 

# Divgan + Random Crop (student trained from scratch on 0.5*Proxy + 0.5*DivGAN )
CUDA_VISIBLE_DEVICES=0 python ./code/train_student/train_student.py --div_gan_netG ./cifar100_run_models/alexnet/out_step2_0_10/synthetic_grey/2/netG_epoch_99.pth --dcgan_netG ./cifar100_run_models/alexnet/synthetic_grey/dcgan/netG_epoch_199.pth  --count 2 --network alexnet --pad_crop  --div_gan_data --div_gan_data_ratio 0.5 --proxy_data --proxy_data_ratio 0.5 --from_scratch --name from_scratch_div_gan_05_synthetic_grey_imgs  --val_data_dcgan ./val_data/dcgan_val_data_synthetic_alexnet_grey.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --proxy_dataset synthetic --true_dataset cifar10 --num_classes 10 --synthetic_dir ./synthetic_dataset/50k_samples/ --total_synth_samples 50000 --grey_scale


CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --div_gan_netG ./cifar100_run_models/alexnet/out_step2_0_10/synthetic_grey/2/netG_epoch_99.pth --network alexnet --div_gan_out ./val_data/degan_val_data_synthetic_alexnet_grey.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth

# ./train_student/checkpoint/alexnet/cifar10_synthetic_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_synthetic_grey_imgs.pth


CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_generator_clone.py --dataroot ./data/  --dataset synthetic --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/out_step2/synthetic_grey/0_500/ --manualSeed 108 --niter 150 --batchSize 64 --netG  ./cifar100_run_models/alexnet/out_step2_0_10/synthetic_grey/2/netG_epoch_99.pth  --student_path ./train_student/checkpoint/alexnet/cifar10_synthetic_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_synthetic_grey_imgs.pth --network alexnet --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth  --warmup --name warmup_altr_0_10_p10  --auto-augment --c_l 0 --d_l 500  --val_data_dcgan ./val_data/dcgan_val_data_synthetic_alexnet_grey.pkl --val_data_degan ./val_data/degan_val_data_synthetic_alexnet_grey.pkl --true_dataset cifar10 --num_classes 10 --synthetic_dir ./synthetic_dataset/50k_samples/  --total_synth_samples 50000 --grey_scale
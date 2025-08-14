Official code for AnimateAnyMesh(ICCV2025)

### Train DyMeshVAE
python train_dvae_dis.py --data_dir /mnt/nas_jianchong/wzj/Obj4d_AMASS_DT4D_All_Data/obj4d_amass_dt4d_all_4096_merged_filtered_16f_clips_caption_no_pad_f_test --val_data_dir /mnt/nas_jianchong/wzj/objaverse-xl-animes/objxl_valset --train_epoch 2000 --batch_size 64 --validate --is_training --lr 4e-4 --avg_loss --exp test_dvae

### Test DyMeshVAE
python test_dvae.py --dataset test --exp test_dvae --epoch 1  --render

### Test DyMeshVAE Factors
<!-- torchrun --standalone --nproc_per_node=4 \ -->
python test_vae_factor.py --data_dir /mnt/nas_jianchong/wzj/Obj4d_AMASS_DT4D_All_Data/obj4d_amass_dt4d_all_4096_merged_filtered_16f_clips_caption_no_pad_f_test --vae_exp test_dvae --vae_epoch 1 --latent_dim 32 --num_t 16

### Train RF Model
python train_diff_dis.py --batch_size 64 --vae_exp test_dvae --vae_epoch 1 --rescale --base_name 40m --data_dir /mnt/nas_jianchong/wzj/Obj4d_AMASS_DT4D_All_Data/obj4d_amass_dt4d_all_4096_merged_filtered_16f_clips_caption_no_pad_f_test --max_length 4096 --train_epoch 2000 --lr 2e-4 --exp test_rf

### Test Drive
python test_drive.py --rescale --num_traj 512 --rf_exp test_rf --rf_epoch 1 --prompt "The object is dancing" --test_name girl --seed 666 --azi 0 --ele 0
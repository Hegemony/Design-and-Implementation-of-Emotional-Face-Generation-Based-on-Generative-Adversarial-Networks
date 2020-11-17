from solver0 import *

def restore_model(self, resume_iters):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(resume_iters))
    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
    D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
    self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

def test_result(self, model, dataset, input):
    """Translate images using StarGAN trained on a single dataset."""
    # Load the trained generator.
    self.restore_model(self.test_iters)

    # Set data loader.
    if self.dataset == 'CelebA':
        data_loader = self.celeba_loader
    elif self.dataset == 'RaFD':
        data_loader = self.rafd_loader

    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):

            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            # Translate images.
            x_fake_list = [x_real]
            for c_trg in c_trg_list:
                x_fake_list.append(self.G(x_real, c_trg))

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))
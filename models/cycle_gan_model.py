import torch
import torch.nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from GPUtil import showUtilization as gpu_usage

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_seg_A', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_seg_B', type=float, default=1, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_seg_from_syn', type=float, default=0, help='use to teach segmentor from synthetic data')
            parser.add_argument('--lambda_gen_from_seg', type=float, default=5, help='use to teach generator from segmentation data')
            parser.add_argument('--first_gen_from_seg_epoch', type=float, default=100, help='first epoch lambda_gen_from_seg is not 0')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.phase = self.opt.train_phase if self.opt.isTrain else 0
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.phase != 1:
            self.loss_names = []
            visual_names_A = ['real_A', 'fake_B', 'ground_truth_seg_A']
            visual_names_B = ['real_B', 'fake_A', 'ground_truth_seg_B']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            if self.isTrain :
                if self.phase != 1: # In phase 1 no Generator and Discriminator
                    self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'GS_A', 'GS_B']
                    self.loss_names.append('rec_A')
                    self.loss_names.append('rec_B')
                    visual_names_A.append('rec_A')
                    visual_names_B.append('rec_B')
                if self.opt.lambda_identity > 0.0 :  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
                    visual_names_A.append('idt_B')
                    visual_names_B.append('idt_A')
                if self.phase == 1 or self.phase == 3:
                    self.loss_names.append('S_A')
                    self.loss_names.append('S_B')
                    visual_names_A.append('seg_A')
                    visual_names_B.append('seg_B')
                if self.phase == 3 :
                    self.loss_names.append('S_SYN_A')
                    self.loss_names.append('S_SYN_B')                
                    visual_names_A.append('seg_fake_B')
                    visual_names_B.append('seg_fake_A')
            else: # Test segmentation
                    self.loss_names.append('S_A')
                    self.loss_names.append('S_B')
                    visual_names_A.append('seg_A')
                    visual_names_B.append('seg_B')
        else: # Phase 1
            self.loss_names = ['S_A', 'S_B']
            visual_names_A = ['real_A', 'ground_truth_seg_A', 'seg_A']
            visual_names_B = ['real_B', 'ground_truth_seg_B', 'seg_B']


        if(self.opt.four_labels):
            self.labels_translate = {
                0 : 0,      # Background
                205 : 1,    # MYO
                420 : 2,    # LAC
                500 : 3,    # LVC
                550 : 0,    # RAC
                600 : 0,    # RVC
                820 : 4,    # AA
                850 : 0,    # pulmonary artery
            }
            self.num_of_labels = 5
        else:
            self.labels_translate = {
                0 : 0,      # Background
                205 : 1,    # MYO
                420 : 2,    # LAC
                500 : 3,    # LVC
                550 : 4,    # RAC
                600 : 5,    # RVC
                820 : 6,    # AA
                850 : 7,    # pulmonary artery
            }
            self.num_of_labels = 8

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain and not self.phase == 1:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'S_A', 'S_B']
        else:  # during test time, only Ss
            self.model_names = ['S_A', 'S_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)


        self.netS_A =  networks.define_S(opt.input_nc, opt.seg_output_nc, opt.nsf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS_B =  networks.define_S(opt.input_nc, opt.seg_output_nc, opt.nsf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and not self.phase == 1:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if not self.phase == 1:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)

            self.optimizer_S = torch.optim.Adam(itertools.chain(self.netS_A.parameters(), self.netS_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'cttomr'

        # When using 1 batch size to hack 2D network:
        # self.real_A = torch.from_numpy(np.rollaxis(input['ct' if AtoB else 'mr'].squeeze(0).numpy(), -1, 0)).to(self.device)
        # self.real_B = torch.from_numpy(np.rollaxis(input['mr' if AtoB else 'ct'].squeeze(0).numpy(), -1, 0)).to(self.device)
        self.real_A = input['ct' if AtoB else 'mr'].to(self.device, dtype=torch.float)
        self.ground_truth_seg_A = input['ct_label' if AtoB else 'mr_label'].to(self.device, dtype=torch.float)
        self.real_B = input['mr' if AtoB else 'ct'].to(self.device, dtype=torch.float)       
        self.ground_truth_seg_B = input['mr_label' if AtoB else 'ct_label'].to(self.device, dtype=torch.float)       
        self.image_paths = input['ct_paths' if AtoB else 'mr_paths']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.phase != 1:
            self.fake_B = self.netG_A(self.real_A)      # G_A(A)
            self.fake_A = self.netG_B(self.real_B)      # G_B(B)
            self.seg_fake_B = self.netS_B(self.fake_B)  # S{G_A(A), Y_A}
            # self.seg_rec_A = self.netS_A(self.rec_A)    # S{G_B(G_A(A)), Y_A}
            self.seg_fake_A = self.netS_A(self.fake_A)  # S{G_B(B), Y_B}
            # self.seg_rec_B = self.netS_B(self.rec_B)    # S{G_A(G_B(B)), Y_A}
        
        if not self.opt.isTrain or self.phase != 2:
            self.seg_A = self.netS_A(self.real_A)
            self.seg_B = self.netS_B(self.real_B)
        else:
            self.rec_A = self.netG_B(self.fake_B)       # G_B(G_A(A))
            self.rec_B = self.netG_A(self.fake_A)       # G_A(G_B(B))



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, epoch):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_seg_A = self.opt.lambda_seg_A
        lambda_seg_B = self.opt.lambda_seg_B

        if epoch < self.opt.first_gen_from_seg_epoch or self.phase != 2:
            lambda_seg = 0
        else:
            lambda_seg = self.opt.lambda_gen_from_seg
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        #Segmentation loss:
        if lambda_seg > 0:
            # with torch.no_grad():
            self.loss_GS_A = self.seg_loss(self.seg_fake_A, self.ground_truth_seg_B) * lambda_seg_A * lambda_seg
            self.loss_GS_B = self.seg_loss(self.seg_fake_B, self.ground_truth_seg_A) * lambda_seg_B * lambda_seg
            # FIXME : Not comapring to the seg feedforward because of high memory consumption running Segmentors
            # self.loss_GS_A = self.seg_loss(self.netS_A(self.fake_A).detach(), self.seg_B) * lambda_seg_A * lambda_seg
            # self.loss_GS_B = self.seg_loss(self.netS_B(self.fake_B).detach(), self.seg_A) * lambda_seg_B * lambda_seg
        else:
            self.loss_GS_A = 0
            self.loss_GS_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_GS_A + self.loss_GS_B
        self.loss_G.backward()

    def seg_loss(self, seg, truth):
        loss = torch.nn.CrossEntropyLoss()
        return loss(seg.permute(0, 2, 3, 4, 1).contiguous().view(-1,8), truth.view(-1,1).squeeze().contiguous().view(-1).long())

    def backward_S(self):
        """Calculate the loss for segmentors S_A and S_B"""
        lambda_seg_from_syn = self.opt.lambda_seg_from_syn if self.phase == 3 else 0

        self.loss_S_A = self.seg_loss(self.seg_A, self.ground_truth_seg_A)
        self.loss_S_B = self.seg_loss(self.seg_B, self.ground_truth_seg_B)
        # Syntheric loss
        if lambda_seg_from_syn > 0:
            self.loss_S_SYN_A = self.seg_loss(self.seg_fake_A, self.ground_truth_seg_B) * lambda_seg_from_syn
            self.loss_S_SYN_B = self.seg_loss(self.seg_fake_B, self.ground_truth_seg_A) * lambda_seg_from_syn
        else:
            self.loss_S_SYN_A = 0
            self.loss_S_SYN_B = 0

        self.loss_S = self.loss_S_A + self.loss_S_B + self.loss_S_SYN_A + self.loss_S_SYN_B
        # combined loss and calculate gradients
        self.loss_S.backward()


    def get_segmentation_by_max(self):
        self.seg_A = self.seg_A.argmax(dim=1, keepdim=True)
        self.seg_B = self.seg_B.argmax(dim=1, keepdim=True)
        if self.phase != 1:
            self.seg_fake_B = self.seg_fake_B.argmax(dim=1, keepdim=True)
            self.seg_fake_A = self.seg_fake_A.argmax(dim=1, keepdim=True)


    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # torch.autograd.set_detect_anomaly(True)

        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        if self.phase == 2:
            # G_A and G_B
            self.set_requires_grad([self.netS_A, self.netS_B], False)  # Ss require no gradients when optimizing Gs and Ds
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G(epoch)             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights      
        else:
            # S_A and S_B
            self.set_requires_grad([self.netS_A, self.netS_B], True) 
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Ss
            self.set_requires_grad([self.netG_A, self.netG_B], False)  # Gs require no gradients when optimizing Ss
            self.optimizer_S.zero_grad()  # set S_A and S_B's gradients to zero
            self.backward_S()             # calculate gradients for S_A and S_B
            self.optimizer_S.step()       # update S_A and S_B's weights
            self.get_segmentation_by_max() #change the segmentation to (B,1,H,W,D) instead (B,8,H,W,D)


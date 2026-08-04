"""
build_model.py
================
InverseModel (generator) + Discriminator, verbatim from every notebook's section 3
(all 11 notebooks are byte-identical here).

The conv layer used by both is an ablation knob (`conv_type`, see config.py):
'plain' / 'deform' are the original two notebook variants; 'dcnv4' / 'involution' /
'dynamic' / 'odconv' are optional stronger replacements implemented in conv_layers.py.
"""
import torch
import torch.nn as nn

from conv_layers import CONV_LAYER_REGISTRY, DeformableConv2d


class InverseModel(torch.nn.Module):
    def __init__(self, conv_layer=DeformableConv2d, num_classes=80, num_filters=32, input_size=(32, 1, 60, 60)):
        super(InverseModel, self).__init__()
        # Decoder
        kernel_size = (5, 5)
        # padding1 = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        ConvLayer = conv_layer  # ablation knob: DeformableConv2d (proposed) or PlainConv2d
        self.decoder_fc1_linear1 = torch.nn.Linear(80, 2400)
        self.decoder_layer1_conv = ConvLayer(150, 3 * num_filters, (5, 5), (1, 1), padding='same')
        self.decoder_layer1_batch_norm = torch.nn.BatchNorm2d(3 * num_filters, eps=1e-3, momentum=0.99)
        # self.decoder_layer1_pooling = torch.nn.MaxUnpool2d((2, 2))
        self.decoder_layer2_conv = ConvLayer(3 * num_filters, 2 * num_filters, (5, 5), (1, 1), padding='same')
        self.decoder_layer2_batch_norm = torch.nn.BatchNorm2d(2 * num_filters, eps=1e-3, momentum=0.99)
        self.decoder_layer3_conv = ConvLayer(2 * num_filters, num_filters, (5, 5), (1, 1), padding='same')
        self.decoder_layer3_batch_norm = torch.nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.99)
        self.activation = torch.nn.ReLU()
        self.activation1 = torch.nn.ReLU()
        self.decoder_layer4_conv = ConvLayer(num_filters, 1, (5, 5), (1, 1), padding='same')
        self.decoder_layer1_pooling = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_layer1_pooling_2 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.linear2 = torch.nn.Linear(40, 40)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(2)
        x = self.decoder_fc1_linear1(x)
        x = x.view((-1, 150, 4, 4))

        x = self.decoder_layer1_conv(x)
        x = self.activation(x)
        x1 = self.decoder_layer1_pooling(x)
        x = self.decoder_layer1_batch_norm(x1)

        x = self.decoder_layer2_conv(x)
        x = self.activation(x)
        x1 = self.decoder_layer1_pooling(x)
        x = self.decoder_layer2_batch_norm(x1)

        x = self.decoder_layer3_conv(x)
        x = self.activation(x)
        x1 = self.decoder_layer1_pooling_2(x)
        x = self.decoder_layer4_conv(x1)
        decoded_output = self.activation(x)
        return decoded_output


class Discriminator(nn.Module):
    def __init__(self, conv_layer=DeformableConv2d):

        super(Discriminator, self).__init__()
        Conv2d = conv_layer  # ablation knob: DeformableConv2d (proposed) or PlainConv2d
        self.model = nn.Sequential(
            Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            # nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


def build_models(conv_type: str, use_gan: bool, device: torch.device):
    """Ablation knob: which ConvLayer (see conv_layers.py / CONV_LAYER_REGISTRY) generator
    and discriminator use. 'deform' reproduces every notebook's section 6
    (`conv_layer = DeformableConv2d if FLAGS['use_deform'] else PlainConv2d`); the other
    types are optional stronger replacements, all sharing the same drop-in signature."""
    conv_layer = CONV_LAYER_REGISTRY[conv_type]
    model = InverseModel(conv_layer=conv_layer).to(device)
    discriminator = Discriminator(conv_layer=conv_layer).to(device) if use_gan else None
    return model, discriminator

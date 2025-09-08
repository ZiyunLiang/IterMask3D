import torch
import torch.nn as nn
from Nets.updated_res_units import ResidualUnit, Convolution

class UNET3D(nn.Module):

	def __init__(self,
	             in_channels: int,
	             out_channels: int = 1,
	             last_layer_conv_only: bool = True
	             ) -> None:
		super().__init__()

		print("3DUNET INIT")
		dropout = 0.2
		print("Dropout: ", dropout)
		normalization = 'BATCH'
		self.conv_1 = ResidualUnit(spatial_dims=3, in_channels=in_channels, out_channels=16, strides=2, kernel_size=3,
		                           subunits=2, dropout=0.2, norm=normalization)
		self.conv_2 = ResidualUnit(spatial_dims=3, in_channels=16, out_channels=32, strides=2, kernel_size=3,
		                           subunits=2, dropout=0.2, norm=normalization)
		self.conv_3 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3,
		                           subunits=2, dropout=0.2, norm=normalization)
		self.conv_4 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3,
		                           subunits=2, dropout=0.2, norm=normalization)
		self.conv_5 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=1, kernel_size=3,
		                           subunits=2, dropout=0.2, norm=normalization)

		upsample = torch.nn.Upsample(scale_factor=2)

		up_conv_1_a = Convolution(spatial_dims=3, in_channels=384, out_channels=384, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_2_a = Convolution(spatial_dims=3, in_channels=128, out_channels=128, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_3_a = Convolution(spatial_dims=3, in_channels=64, out_channels=64, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_4_a = Convolution(spatial_dims=3, in_channels=32, out_channels=32, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)

		up_conv_1_b = Convolution(spatial_dims=3, in_channels=384, out_channels=64, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_2_b = Convolution(spatial_dims=3, in_channels=128, out_channels=32, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_3_b = Convolution(spatial_dims=3, in_channels=64, out_channels=16, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization)
		up_conv_4_b = Convolution(spatial_dims=3, in_channels=32, out_channels=out_channels, strides=1, kernel_size=3,
		                          dropout=0.2, norm=normalization, conv_only=last_layer_conv_only)

		self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
		self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
		self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
		self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		conv_out_1 = self.conv_1(x)
		conv_out_2 = self.conv_2(conv_out_1)
		conv_out_3 = self.conv_3(conv_out_2)
		conv_out_4 = self.conv_4(conv_out_3)
		conv_out_5 = self.conv_5(conv_out_4)

		up_in_1 = torch.cat((conv_out_5, conv_out_4), dim=1)
		up_out_1 = self.up_stage_1(up_in_1)

		up_in_2 = torch.cat((up_out_1, conv_out_3), dim=1)
		up_out_2 = self.up_stage_2(up_in_2)

		up_in_3 = torch.cat((up_out_2, conv_out_2), dim=1)
		up_out_3 = self.up_stage_3(up_in_3)

		up_in_4 = torch.cat((up_out_3, conv_out_1), dim=1)
		up_out_4 = self.up_stage_4(up_in_4)

		return up_out_4

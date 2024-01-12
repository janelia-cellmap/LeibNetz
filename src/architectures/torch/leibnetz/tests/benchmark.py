# %%
import torch
from architectures.torch.leibnetz.unet import build_unet as leibnetz_unet
from funlib.learn.torch.models import UNet as funlib_unet
# set flag to improve training speeds
torch.backends.cudnn.benchmark = True

def benchmark():
    l_unet = leibnetz_unet()
    l_unet_c = leibnetz_unet().trace()
    
    f_unet = funlib_unet(
        downsample_factors=[(2, 2, 2), (2, 2, 2)],
        kernel_size_down=[[(3, 3, 3), (3, 3, 3), (3, 3, 3)],]*3,
        kernel_size_up=[[(3, 3, 3), (3, 3, 3), (3, 3, 3)],]*3,
        in_channels=1,
        num_fmaps_out=1,
        num_fmaps=12,
        fmap_inc_factor=2,)
    # f_unet_c = torch.compile(funlib_unet(
    #     downsample_factors=[(2, 2, 2), (2, 2, 2)],
    #     kernel_size_down=[[(3, 3, 3), (3, 3, 3), (3, 3, 3)],]*3,
    #     kernel_size_up=[[(3, 3, 3), (3, 3, 3), (3, 3, 3)],]*3,
    #     in_channels=1,
    #     num_fmaps_out=1,
    #     num_fmaps=12,
    #     fmap_inc_factor=2,))

    def get_input_dict(device):
        inputs = {}
        for k, v in l_unet.input_shapes.items():
            inputs[k] = torch.rand(
                (
                    1,
                    1,
                )
                + tuple(v[0].astype(int))
            ).to(device)
        return inputs
    
    def get_input(device):
        return torch.rand((1,1,)+tuple(l_unet.input_shapes["input"][0].astype(int))).to(device)

    def benchmark_cpu():
        nonlocal get_input_dict, get_input
        l_unet.cpu()
        l_unet_c.cpu()
        f_unet.cpu()
        # f_unet_c.cpu()
        print("_____________________________________________________")
        print("On CPU:")
        print("_____________________________________________________")
        print()
        print("Leibnetz UNet")
        print("\tNormal:")
        inputs = get_input_dict("cpu")
        %timeit l_unet(inputs)
        print("\tTraced:")
        %timeit l_unet_c(inputs)
        print()
        print("Funlib UNet")
        # print("\tNormal:")
        inputs = get_input("cpu")
        %timeit f_unet(inputs)
        # print("\tCompiled:")
        # %timeit f_unet_c(inputs)

    def benchmark_gpu():
        nonlocal get_input_dict, get_input
        l_unet.cuda()
        l_unet_c.cuda()
        f_unet.cuda()
        # f_unet_c.cuda()
        print("_____________________________________________________")
        print("On GPU:")
        print("_____________________________________________________")
        print()
        print("Leibnetz UNet")
        print("\tNormal:")
        inputs = get_input_dict("cuda")
        %timeit l_unet(inputs)
        print("\tTraced:")
        %timeit l_unet_c(inputs)
        print()
        print("Funlib UNet")
        # print("\tNormal:")
        inputs = get_input("cuda")
        %timeit f_unet(inputs)
        # print("\tCompiled:")
        # %timeit f_unet_c(inputs)

    def benchmark_eval():
        l_unet.eval()
        l_unet_c.eval()
        f_unet.eval()
        # f_unet_c.eval()
        print("=====================================================")
        print("EVAL MODE")
        print("=====================================================")
        benchmark_cpu()
        benchmark_gpu()

    def benchmark_train():
        l_unet.train()
        l_unet_c.train()
        f_unet.train()
        # f_unet_c.train()
        print("=====================================================")
        print("TRAIN MODE")
        print("=====================================================")
        benchmark_cpu()
        benchmark_gpu()

    benchmark_eval()
    print()
    print()
    benchmark_train()

# %%
if __name__ == "__main__":
    benchmark()
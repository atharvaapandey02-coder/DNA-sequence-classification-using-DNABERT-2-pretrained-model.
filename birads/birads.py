import torch
import utils
import models_torch as models

def inference(parameters, verbose=True):
    """
    Function that creates a model, loads the parameters, and makes a prediction
    """
    device = torch.device(
        f"cuda:{parameters['gpu_number']}" if parameters["device_type"] == "gpu"
        else "cpu"
    )

    # Initialize and load model
    model = models.BaselineBreastModel(device, nodropout_probability=1.0,
                                     gaussian_noise_std=0.0).to(device)
    model.load_state_dict(torch.load(parameters["model_path"]))
    model.eval()

    # Load and prepare images
    x = {}
    for view in ["L-CC", "R-CC", "L-MLO", "R-MLO"]:
        image = utils.load_images(parameters['image_path'], view,
                                parameters['input_size'])
        x[view] = torch.Tensor(image).permute(0, 3, 1, 2).to(device)

    # Run prediction
    with torch.no_grad():
        prediction = model(x).cpu().numpy()[0]

    if verbose:
        print('BI-RADS prediction:')
        for i, prob in enumerate(prediction):
            print(f'\tBI-RADS {i}:\t{prob}')

    return prediction

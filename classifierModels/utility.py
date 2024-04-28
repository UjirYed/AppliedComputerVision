from prettytable import PrettyTable
import torch


def count_parameters(model):
    """
    A useful function that takes in a model, and returns a table containing a summary of all parameters, trainable or not.
    Modified from SO: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    """
    table = PrettyTable(["Modules", "Total Params", "Total Trainable Params"])
    total_params = 0
    total_train_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            table.add_row([name, params, params])
            total_params += params
            total_train_params += params
        else:
            table.add_row([name, params, 0])
            total_params += params
    print(table)
    print(f"Total Params: {total_params}, Total Trainable Params: {total_train_params}")
    return total_params, total_train_params
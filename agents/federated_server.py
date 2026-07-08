import torch
import copy

class FederatedServer:
    def aggregate_weights(self, local_weights_list):
        if not local_weights_list: return None
        global_weights = copy.deepcopy(local_weights_list[0])
        for key in global_weights.keys():
            for i in range(1, len(local_weights_list)):
                global_weights[key] += local_weights_list[i][key]
            global_weights[key] = torch.div(global_weights[key], len(local_weights_list))
        return global_weights

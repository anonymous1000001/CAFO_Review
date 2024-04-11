import torch
import torch.nn as nn


class CustomDropOut(nn.Module):
    def __init__(self, cfg):
        super(CustomDropOut, self).__init__()
        self.cfg = cfg
        self.dropout_rate = cfg.predict.dropout_rate
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError("Dropout rate must be between 0 and 1")

    def set_dropout_rate(self, dropout_rate):
        assert dropout_rate >= 0 and dropout_rate <= 1, "Dropout rate must be between 0 and 1"
        self.dropout_rate = dropout_rate
        print(f"Dropout rate set to {self.dropout_rate}")

    def get_dropout_rate(self):
        return self.dropout_rate

    def forward(self, channel_attention_score):
        assert channel_attention_score.dim() == 4, "Input must be 4D tensor, (batch, channel, height, width)"
        if self.training == True:
            raise ValueError("This dropout layer is only for predict mode. The error should not be raised!")
        else:
            """Use this only for predict mode, This is different from test mode."""
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.dropout_rate)
            drop_mask = binomial.sample(channel_attention_score.shape).to(channel_attention_score)
            return channel_attention_score * drop_mask, drop_mask


class ChannelDropOut(nn.Module):
    def __init__(self, cfg):
        super(ChannelDropOut, self).__init__()
        self.cfg = cfg
        self.alpha_param = cfg.beta_dropout.alpha_param * torch.ones(cfg.task.in_channels)
        self.beta_param = cfg.beta_dropout.beta_param * torch.ones(cfg.task.in_channels)
        self.explore_epoch = cfg.beta_dropout.explore_epoch
        self.reward_value = cfg.beta_dropout.reward_value
        print(f"Initialized!")

    def get_alpha_beta(self):
        # return a copy of the alpha and beta parameters.
        return self.alpha_param.clone(), self.beta_param.clone()

    def forward(self, X, current_epoch=-1):
        assert X.dim() == 4, "Input must be 4D tensor, (batch, channel, height, width)"

        if (self.training == True) and (current_epoch < self.explore_epoch):
            ############ EXPLORATION ############
            return X
        elif (self.training == True) and (current_epoch >= self.explore_epoch):
            ############# EXPLOITATION #############
            mean_value = X.mean(dim=0).squeeze()
            beta_dist_dict = {}
            binomial_dict = {}
            beta_proba_list = []
            mask_proba_list = []

            for idx, gi_score in enumerate(mean_value):

                # sample a binomial probability from the beta distribution.
                beta_dist_dict[idx] = torch.distributions.beta.Beta(self.alpha_param[idx], self.beta_param[idx])
                beta_proba = beta_dist_dict[idx].sample()
                beta_proba_list.append(beta_proba.item())
                binomial_dict[idx] = torch.distributions.binomial.Binomial(probs=beta_proba)
                mask_proba_list.append(binomial_dict[idx].sample())

                # update
                if gi_score >= 0.5:
                    reward = self.reward_value
                else:
                    reward = 0

                self.alpha_param[idx] += reward
                self.beta_param[idx] += self.reward_value - reward
            mask_proba = torch.stack(mask_proba_list, dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask_proba = mask_proba.to(X)
            print(f"Channel Dropout Keeping Portion: {100*torch.mean(mask_proba):.2f}% from the full attention, ")
            mask = mask_proba.expand(X.size())
            return X * mask, mask_proba

        elif self.training == False:
            # get mean of the beta distribution.
            # return X
            mean_values = self.alpha_param / (self.alpha_param + self.beta_param)
            mask = []
            for idx, mean_value in enumerate(mean_values):
                if mean_value >= 0.5:
                    mask.append(1)
                else:
                    mask.append(0)
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask = mask.to(X)
            mask = mask.expand(X.size())
            return X * mask


# class ChannelDropOut(nn.Module):
#     def __init__(self, cfg):
#         super(ChannelDropOut, self).__init__()
#         self.cfg = cfg
#         self.alpha_param = cfg.beta_dropout.alpha_param * torch.ones(cfg.task.in_channels)
#         self.beta_param = cfg.beta_dropout.beta_param * torch.ones(cfg.task.in_channels)

#         print(f"Initialized!")

#     def forward(self, X):
#         assert X.dim() == 4, "Input must be 4D tensor, (batch, channel, height, width)"
#         if self.training:
#             mean_value = X.mean(dim=0).squeeze()
#             beta_dict = {}
#             binomial_dict = {}
#             beta_proba_list = []
#             mask_proba_list = []

#             for idx, gi_score in enumerate(mean_value):

#                 # sample a binomial probability from the beta distribution.
#                 beta_dict[idx] = torch.distributions.beta.Beta(gi_score, torch.tensor([self.beta_param]).to(gi_score))
#                 beta_proba = beta_dict[idx].sample()
#                 beta_proba_list.append(beta_proba.item())
#                 binomial_dict[idx] = torch.distributions.binomial.Binomial(probs=beta_proba)
#                 mask_proba_list.append(binomial_dict[idx].sample())
#             mask_proba = torch.stack(mask_proba_list, dim=1).unsqueeze(-1).unsqueeze(-1)
#             print(f"Mask proba: {torch.mean(mask_proba)}, ")
#             mask = mask_proba.expand(X.size())
#             return X * mask, mask_proba
#         return X

import torch
import torch.nn as nn
import torch.nn.functional as F

class Criterion(nn.Module):
    """ Class for managing loss computation.

    """

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Compute the loss. Subclass must override this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        Returns:
            A non-reduced FloatTensor with shape (batch, )
        """
        raise NotImplementedError

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
        """
        Compute loss given inputs and labels.

        Args:
            inputs: Input tensor of the criterion.
            labels: Label tensor of the criterion.
            reduce: Boolean value indicate whether the criterion should reduce the loss along the batch. If false,
                the criterion return a FloatTensor with shape (batch, ), otherwise a scalar.
            normalization: Normalization factor of the loss. Should be a float scalar or a FloatTensor with shape
                (batch, )
        """
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss


class NMTCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, padding_idx, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='none')

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels, **kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss
class SilentCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, label_smoothing=0.0):

        super().__init__()


        self.criterion = F.mse_loss()


    def _compute_loss(self, inputs, labels, **kwargs):


        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss

class CtcCriterion(Criterion):
    def __init__(self,blank_id,reduction='mean',):
        super().__init__()
        self.blank_id = blank_id
        self.ctc_criterion = nn.CTCLoss(blank=self.blank_id,reduction='none')
    
    def _compute_loss(self,inputs,labels,input_lengths,target_lengths):
        ctc_loss = self.ctc_criterion(inputs,labels,input_lengths,target_lengths)

        return ctc_loss

class JointCtcAttention(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self,blank_id, padding_idx, label_smoothing=0.0,weight=0.5,reduction='mean'):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        self.blank_id = blank_id
        self.weight = weight
        self.ctc_criterion = nn.CTCLoss(blank=self.blank_id,reduction='none')

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='none')

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels,encoder_output,ctc_targets,input_lengths,target_lengths,**kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        att_loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)
        ctc_loss = self.ctc_criterion(encoder_output,ctc_targets,input_lengths,target_lengths)
        loss = self.weight*att_loss+(1-self.weight)*ctc_loss

        return loss        

class JointCtcDekayAttention(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self,blank_id, padding_idx, label_smoothing=0.0,reduction='mean'):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        self.blank_id = blank_id
        
        self.ctc_criterion = nn.CTCLoss(blank=self.blank_id,reduction='none')

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='none')

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels,encoder_output,ctc_targets,input_lengths,target_lengths,weight,**kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()
        att_loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        ctc_loss = self.ctc_criterion(encoder_output,ctc_targets,input_lengths,target_lengths)
        
        loss = weight*att_loss + (1 - weight)*ctc_loss

        return loss        




class VoiceJointCtcAttention(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self,blank_id, padding_idx, label_smoothing=0.0,reduction='mean'):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        self.blank_id = blank_id
        
        self.ctc_criterion = nn.CTCLoss(blank=self.blank_id,reduction='none')

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='none')

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels,encoder_output,ctc_targets,input_lengths,target_lengths,weight, voice,**kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()


        att_loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)
        ctc_loss = self.ctc_criterion(encoder_output,ctc_targets,input_lengths,target_lengths)
        


        loss = weight*torch.mul(voice,att_loss) + (1 - weight)*torch.mul(voice,ctc_loss)

        return loss    







class MSECriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self,inputs,target):
        loss = self.criterion(inputs,target)

        return loss



class TransducerLoss(nn.Module):
    """
    Transducer loss module.
    Args:
        blank_id (int): blank symbol id
    """

    def __init__(self, blank_id):
        """Construct an TransLoss object."""
        super().__init__()
        try:
            from warprnnt_pytorch import rnnt_loss
        except ImportError:
            raise ImportError("warp-rnnt is not installed. Please re-setup")
        self.rnnt_loss = rnnt_loss
        self.blank_id = blank_id

    def forward(
            self,
            log_probs: torch.FloatTensor,
            targets: torch.IntTensor,
            input_lengths: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Compute path-aware regularization transducer loss.
        Args:
            log_probs (torch.FloatTensor): Batch of predicted sequences (batch, maxlen_in, maxlen_out+1, odim)
            targets (torch.IntTensor): Batch of target sequences (batch, maxlen_out)
            input_lengths (torch.IntTensor): batch of lengths of predicted sequences (batch)
            target_lengths (torch.IntTensor): batch of lengths of target sequences (batch)
        Returns:
            loss (torch.FloatTensor): transducer loss
        """

        return self.rnnt_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
            blank=self.blank_id,
        )
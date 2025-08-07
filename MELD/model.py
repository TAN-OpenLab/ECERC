import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.args = args
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + 0. * self.pos_table[:, :x.size(1), :x.size(2)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [b x n x lq x d_k], [b x n x lq x d_k], [b x n x lq x d_v]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model * 2, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        enc_output = self.slf_attn(q, k, v, mask=mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layer, n_head, d_model, d_k, d_v, dropout=0.1, scale_emb=False):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_head, d_model, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)])
        # self.layer_norm_q = nn.LayerNorm(d_model, eps=1e-6)
        # self.layer_norm_k = nn.LayerNorm(d_model, eps=1e-6)
        # self.layer_norm_v = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, q, k, v, mode, mask=None):

        # -- Forward
        if self.scale_emb:
            q *= self.d_model ** 0.5
            k *= self.d_model ** 0.5
            v *= self.d_model ** 0.5
        for idx, enc_layer in enumerate(self.layer_stack):
            q = enc_layer(q, k, v, mask=mask)
            if mode == "self":
                k = v = q
        return q


class ModalFilterV2(nn.Module):
    ''' An encoder model with self attention mechanism. '''

    def __init__(
            self, d_t, d_a, d_v, hidden):

        super().__init__()
        self.proj_t = nn.Linear(d_t, hidden)
        self.proj_a = nn.Linear(d_a, hidden)
        self.proj_v = nn.Linear(d_v, hidden)

        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)


    def forward(self, t, a, v):
        """

        :param t: (B, L, D_t)
        :param a:  (B, L, D_a)
        :param v:  (B, L, D_v)
        :return:
        """
        t, a, v = self.proj_t(t), self.proj_a(a), self.proj_v(v)
        # t <-- a, v
        w_t = F.sigmoid(self.q(t) + self.k(a) + self.k(v))
        w_a = F.sigmoid(self.q(a) + self.k(t) + self.k(v))
        w_v = F.sigmoid(self.q(v) + self.k(t) + self.k(a))

        return torch.cat([w_t * (t), w_a * (a), w_v * (v)], dim=-1)


class ECERC(nn.Module):
    def __init__(self, args, d_t, d_a, d_v, base_layer=3, input_size=None, hidden_size=None, n_speakers=2,
                 n_classes=7, cuda_flag=False):
        """
        Contextual Reasoning Network
        """

        super(ECERC, self).__init__()
        self.base_layer = base_layer
        self.n_speakers = n_speakers
        self.hidden_size = hidden_size
        self.cuda_flag = cuda_flag
        self.d_t,self.d_a, self.d_v = d_t, d_a, d_v

        self.position_enc_evi = PositionalEncoding(args, d_t+d_a+d_v, n_position=200)

        self.dropout1 = nn.Dropout(p=0.)
        self.dropout2 = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

        self.modal_filtering = ModalFilterV2(d_t, d_a, d_v, hidden_size)
        input_size = hidden_size * 3
        self.emotion_encoding = Encoder(n_layer=base_layer, n_head=8, d_model=input_size, d_k=64, d_v=64)
        self.proj_eve = nn.Linear(d_t, hidden_size)
        self.event_encoding = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)

        self.self_contagion_attention = Encoder(n_layer=base_layer, n_head=8, d_model=input_size, d_k=64, d_v=64)
        self.cross_emotion_attention = Encoder(n_layer=base_layer, n_head=8, d_model=input_size, d_k=64, d_v=64)
        self.self_event_attention = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)
        self.cross_event_attention = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)

        self.gate_reset = nn.Linear(input_size, input_size)
        self.gate_reset2 = nn.Linear(input_size, input_size)
        self.smax_fc = nn.Linear(input_size * 2 + input_size * 2, n_classes)


    def forward(self, U_e, U_s, qmask, umask, seq_lengths):
        # For VAT Modalities.
        # (b,l,d), (b,l,p)
        U_e_, U_s_, qmask_ = U_e.transpose(0, 1),U_s.transpose(0, 1), qmask.transpose(0, 1)
        B, L, D, P = U_e_.size(0), U_e_.size(1), U_e_.size(2), qmask_.size(2)

        U_e_ = self.position_enc_evi(U_e_)
        U_s_ = self.position_enc_evi(U_s_)

        """Construct Masks"""
        # Identity matrix
        identity_mask = torch.eye(umask.size(1)).unsqueeze(0).bool()

        # Subsequence mask.
        submask = (1 - torch.triu(torch.ones((1, umask.size(1), umask.size(1)), device=umask.device), diagonal=1)).bool()
        # Padding mask.
        padmask = umask.unsqueeze(-2).bool()
        # 2->10
        weights = 2 ** torch.arange(P - 1, -1, -1).float()
        weights = weights.cuda() if self.cuda_flag else weights
        decimal_tensor = torch.matmul(qmask_, weights).int()  # (B, L)

        # Self(Speaker2Speaker) mask.
        smask = decimal_tensor
        smask = smask.unsqueeze(1).expand(-1, L, -1) # [B,L,L]
        smask = (smask.transpose(1,2) == smask).bool() # [B,L,L]
        smask = submask & padmask & smask
        smask = smask.cuda() if self.cuda_flag else smask
        # Cross(Speaker2Listener) mask.
        cmask = decimal_tensor
        cmask = cmask.unsqueeze(1).expand(-1, cmask.size(1), -1) # [B,L,L]
        cmask = (cmask.transpose(1,2) != cmask).bool() # [B,L,L]
        cmask = submask & padmask & cmask
        cmask = cmask.cuda() if self.cuda_flag else cmask

        if self.cuda_flag:
            imask, smask, cmask, mask = identity_mask.cuda(), smask.cuda(), cmask.cuda(), (submask & padmask).cuda()

        """Evidence Encoding"""
        evidence_enc = self.modal_filtering(U_e_[:,:,:self.d_t],U_e_[:,:,self.d_t:(self.d_t+self.d_a)],U_e_[:,:,(self.d_t+self.d_a):])  # original
        evidence_enc = self.dropout1(evidence_enc)
        """Emotion Encoding as Causes"""
        emotion_enc = self.emotion_encoding(evidence_enc,evidence_enc,evidence_enc, "self", mask)

        """Event Encoding as Causes"""
        U_s_ = self.proj_eve(U_s_)
        event_enc = self.event_encoding(U_s_,U_s_,U_s_, "self", mask)

        """Cause Fusion: Influenced by Trigger Events Mentioned by the Speaker Self."""
        R_self_event = self.self_event_attention(evidence_enc[:,:,:self.hidden_size], event_enc, event_enc, "cross", imask)
        R_self_event = torch.cat([R_self_event, evidence_enc[:,:,self.hidden_size:]], dim=-1)

        """Cause Fusion: Influenced by Trigger Events Mentioned by Other Speakers."""
        R_cross_event = self.cross_event_attention(evidence_enc[:,:,:self.hidden_size], event_enc, event_enc, "cross", cmask)
        R_cross_event = torch.cat([R_cross_event, evidence_enc[:,:,self.hidden_size:]], dim=-1)

        """Cause Fusion: Self_Contagion."""
        R_contagion = self.self_contagion_attention(evidence_enc, emotion_enc, emotion_enc, "cross", smask & (~imask))  # (B, L, D)

        """Cause Fusion: Influenced by Other Speakers' Emotions."""
        R_cross_emotion = self.cross_emotion_attention(evidence_enc, emotion_enc, emotion_enc, "cross", cmask)

        """GATE"""
        R_lis = [R_contagion, R_cross_emotion,R_self_event,R_cross_event]
        wR_lis= [F.sigmoid(self.gate_reset(R)) * (R) if idx<2 else F.sigmoid(self.gate_reset2(R)) * (R) for idx, R in enumerate(R_lis)]
        R = torch.cat(wR_lis,dim=-1)  # (B, L, D)
        hidden = self.smax_fc(self.dropout2(R))

        log_prob = F.log_softmax(hidden, 2).transpose(0,1)  # (L, B, 4*1024)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob



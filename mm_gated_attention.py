# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMGatedAttention(nn.Module):
    """
    多模态门控注意力机制
    基于GraphCFC中的MMGatedAttention模块，用于融合多模态特征
    """
    
    def __init__(self, mem_dim, cand_dim, att_type='general', dropout=0.5):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        
        # Dropout层
        self.dropouta = nn.Dropout(dropout)
        self.dropoutv = nn.Dropout(dropout)
        self.dropoutl = nn.Dropout(dropout)
        
        if att_type == 'av_bg_fusion':
            # 音频-文本融合
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            # 视觉-文本融合
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == 'general':
            # 单模态变换
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            
            # 双模态门控
            self.transform_av = nn.Linear(mem_dim*3, 1)
            self.transform_al = nn.Linear(mem_dim*3, 1)
            self.transform_vl = nn.Linear(mem_dim*3, 1)
    
    def forward(self, a, v, l, modals=None):
        """
        前向传播
        Args:
            a: 音频特征 [batch_size, seq_len, mem_dim]
            v: 视觉特征 [batch_size, seq_len, mem_dim]
            l: 文本特征 [batch_size, seq_len, mem_dim]
            modals: 使用的模态列表，如['a', 'v', 'l']
        Returns:
            融合后的特征
        """
        if modals is None:
            modals = ['a', 'v', 'l']
        
        # 应用dropout
        a = self.dropouta(a) if len(a) != 0 and 'a' in modals else a
        v = self.dropoutv(v) if len(v) != 0 and 'v' in modals else v
        l = self.dropoutl(l) if len(l) != 0 and 'l' in modals else l
        
        if self.att_type == 'av_bg_fusion':
            outputs = []
            
            # 音频-文本融合
            if 'a' in modals and 'l' in modals:
                fal = torch.cat([a, l], dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa * (self.scalar_al(a))
                outputs.append(hma)
            
            # 视觉-文本融合
            if 'v' in modals and 'l' in modals:
                fvl = torch.cat([v, l], dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv * (self.scalar_vl(v))
                outputs.append(hmv)
            
            # 添加文本特征
            if 'l' in modals:
                outputs.append(l)
            
            # 拼接所有输出
            if len(outputs) > 1:
                hmf = torch.cat(outputs, dim=-1)
            else:
                hmf = outputs[0] if outputs else l
            
            return hmf
            
        elif self.att_type == 'general':
            # 单模态变换
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l
            
            outputs = []
            
            # 音频-视觉融合
            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a*v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 'l' not in modals:
                    return h_av
                outputs.append(h_av)
            
            # 音频-文本融合
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a*l], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * hl
                if 'v' not in modals:
                    return h_al
                outputs.append(h_al)
            
            # 视觉-文本融合
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v*l], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * hl
                if 'a' not in modals:
                    return h_vl
                outputs.append(h_vl)
            
            # 拼接所有双模态融合结果
            if len(outputs) > 1:
                return torch.cat(outputs, dim=-1)
            elif len(outputs) == 1:
                return outputs[0]
            else:
                # 如果没有双模态融合，返回单模态特征的拼接
                single_outputs = []
                if 'a' in modals:
                    single_outputs.append(ha)
                if 'v' in modals:
                    single_outputs.append(hv)
                if 'l' in modals:
                    single_outputs.append(hl)
                return torch.cat(single_outputs, dim=-1) if single_outputs else hl


class MultiModalFusionLayer(nn.Module):
    """
    多模态融合层，集成MMGatedAttention和额外的处理
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, att_type='general'):
        super(MultiModalFusionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 多模态门控注意力
        self.mm_gated_attention = MMGatedAttention(
            mem_dim=input_dim,
            cand_dim=hidden_dim,
            att_type=att_type,
            dropout=dropout
        )
        
        # 根据注意力类型确定融合后的维度
        if att_type == 'general':
            # general模式下，三个双模态融合结果拼接
            fused_dim = hidden_dim * 3
        else:
            # av_bg_fusion模式下的维度
            fused_dim = hidden_dim * 3  # 假设三个模态都存在
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 残差连接的投影层（如果维度不匹配）
        self.residual_projection = None
        if input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, text_features, audio_features, visual_features, modals=['a', 'v', 'l']):
        """
        前向传播
        Args:
            text_features: 文本特征 [batch_size, seq_len, input_dim]
            audio_features: 音频特征 [batch_size, seq_len, input_dim]
            visual_features: 视觉特征 [batch_size, seq_len, input_dim]
            modals: 使用的模态列表
        Returns:
            融合后的特征 [batch_size, seq_len, output_dim]
        """
        # 多模态门控注意力融合
        fused_features = self.mm_gated_attention(
            a=audio_features,
            v=visual_features,
            l=text_features,
            modals=modals
        )
        
        # 输出投影
        output = self.output_projection(fused_features)
        
        # 残差连接（使用文本特征作为残差）
        if self.residual_projection is not None:
            residual = self.residual_projection(text_features)
        else:
            residual = text_features
        
        return output + residual

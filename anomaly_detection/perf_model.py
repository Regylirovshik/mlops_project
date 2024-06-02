import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from datetime import datetime
def sort_batch(data, time_data, label,length):
    batch_size=data.size(0)
    sorted_lengths, inx = torch.sort(length, dim=0, descending=True)
    data = data[inx]
    time_data = time_data[inx]
    label = label[inx]
    length = length[inx]
    length = list(length.cpu().numpy())
    return (data, time_data, label, length)



class TimeEmbedding(nn.Module):
    def __init__(self, batch_size, hidden_embedding_size, output_dim):
        super(TimeEmbedding, self).__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.hidden_embedding_size = hidden_embedding_size
        self.params = nn.ParameterDict({
            'weights': nn.Parameter(torch.randn(1, hidden_embedding_size)),
            'biases': nn.Parameter(torch.rand(hidden_embedding_size)),
            'embedding_matrix': nn.Parameter(torch.rand(hidden_embedding_size, output_dim))
        })
        self.init_weights()
        self.pred = nn.Softmax(dim=1)

        
    def init_weights(self):
        weights = (param.data for name, param in self.named_parameters() if 'weights' in name)
        bias = (param.data for name, param in self.named_parameters() if 'bias' in name)
        embedding_matrix = (param.data for name, param in self.named_parameters() if 'embedding_matrix' in name)
        for k in weights:
            nn.init.xavier_uniform_(k)
        for k in bias:
            nn.init.zeros_(k)
        for k in embedding_matrix:
            nn.init.xavier_uniform_(k)

    def forward(self, x):
        output = []
        x = x.unsqueeze(2)
        projection = torch.mul(x, self.params['weights']) + self.params['biases'] # [h,1]
        s = self.pred(projection)  
        embed = torch.einsum('bsv,vi->bsi', s, self.params['embedding_matrix'])

        return embed



class BiLSTM(nn.Module):
    def __init__(self,batch_size, embed_dim, hidden_dim, num_layers, output_dim, biFlag, perf, attnFlag, time_embedding_dim, device, dropout=0.5):
        
        super(BiLSTM,self).__init__()
        self.perf = perf
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag
        self.dropout = dropout
        self.device = device
        self.attnFlag = attnFlag
        self.time_embedding_dim = time_embedding_dim
        self.input_dim = embed_dim + time_embedding_dim
        self.timeembedding = TimeEmbedding(batch_size, hidden_dim, time_embedding_dim)
        self.layer1 = nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=self.input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=False))
        if(biFlag):
                self.layer1.append(nn.LSTM(input_size=self.input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=False))


        self.layer2=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.LogSoftmax(dim=1)
        )
        if attnFlag:
            self.fc_att = nn.Sequential(
                nn.Linear(hidden_dim * 2, 1),
                nn.Tanh()
                )

        self.to(self.device)

    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_dim, dtype=torch.float32).to(self.device),
                torch.zeros(self.num_layers,batch_size,self.hidden_dim, dtype=torch.float32).to(self.device))
    

    def forward(self,x, t_x,y=None,length=None):
        batch_size=x.size(0)
        max_length=torch.max(length)
        t_x = t_x[:, 0:max_length]
        p = datetime.now()
        
        t_e_x = self.timeembedding(t_x)
        x = x[:, 0:max_length, :]

        y = y[:]
        length = length[:]
         
        x, t_e_x,y,length=sort_batch(x, t_e_x,y,length)
        x_cat = torch.cat([x.type(torch.FloatTensor), t_e_x.type(torch.FloatTensor)], 2)
        hidden=[ self.init_hidden(batch_size) for l in range(self.bi_num)]

        out=[x_cat,reverse_padded_sequence(x_cat,length,batch_first=True)]
        out[0] = out[0].type(torch.FloatTensor).to(self.device)
        out[1] = out[1].type(torch.FloatTensor).to(self.device)
        for l in range(self.bi_num):
            # pack sequence
            out[l]=pack_padded_sequence(out[l],length,batch_first=True)
            out[l],hidden[l]=self.layer1[l](out[l],hidden[l])
            # unpack
            out[l],_=pad_packed_sequence(out[l],batch_first=True)
            # 
            if(l==1):out[l]=reverse_padded_sequence(out[l],length,batch_first=True)
    

        if(self.bi_num==1):out=out[0]
        else:out=torch.cat(out,2)
        # attention
        if self.attnFlag:
            att = self.fc_att(out).squeeze(-1)  # [b,msl,h*2]->[b,msl]
            r_att = torch.sum(att.unsqueeze(-1) * out, dim=1)  # [b,h*2]        
            output = self.layer2(r_att)
        else:
            output = self.layer2(out)
        return y, output, length

def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = torch.tensor(ind.expand_as(inputs))
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

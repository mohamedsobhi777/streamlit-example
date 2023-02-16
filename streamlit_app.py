

import helper_functions as hf

import streamlit as st
import pandas as pd
import numpy as np 
import random

# import os
from pathlib import Path

#Last update: Feb 1st
#Page wide layout
st.set_page_config(layout="wide")


st.header("""
    CryptoPhunks wash trading analysis
""")


st.markdown('---')

st.header("""
Collections Visualization
""")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(104, 149, 499);
    margin-left:40%;
    padding:10px;
    font-size: 1.3em;
}
</style>""", unsafe_allow_html=True)

selected_item = 1
item_id = st.selectbox("Select item id", list(range(1, 3000)), selected_item)

if st.button("Random Item"):
    selected_item = item_id = random.randint(1, 3000)


st.header(f"Current item: {selected_item}")



COLLECTION_ADDRESSES = {
    'cryptopunks': '0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb',
    'cryptophunks': '0xf07468ead8cf26c752c676e43c814fee9c8cf402',
}

lhs , rhs= st.columns(2)
with lhs:
    st.header(f"CryptoPunks (original by Larva Labs)")
    st.image(f"./punks/ETHEREUM/0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb/images/{item_id}.png", width=600)
    # st.write(f"Average Transaction: {0}")

with rhs:
    st.header(f"CryptoPhunks (fake by Not Larva Labs - Inherited)")
    st.image(f"./punks/ETHEREUM/0xf07468ead8cf26c752c676e43c814fee9c8cf402/images/{item_id}.png", width=600)
    # st.write(f"Average Transaction: {0}")



st.markdown("---")

a1, a2, a3, a4, a5 = st.columns(5)
with a3:
    st.header("Sales analysis")

# Load the data 
cryptophunks_activities_fpath = './punks/ETHEREUM/0xf07468ead8cf26c752c676e43c814fee9c8cf402/activities_SELL.json'


# # Read CSV files

cryptophunks_activities = pd.read_json(cryptophunks_activities_fpath)

# # Date info
cryptophunks_activities['year-month'] = cryptophunks_activities.date.apply(lambda x: '-'.join(str(x).split('-')[:2]))


cryptophunks_activities['token_id'] = cryptophunks_activities.nft.apply(lambda x: int(x['type']['tokenId']))


# st.write(cryptophunks_activities)

import matplotlib.pyplot as plt

def plotLineChart(dclasses, ylabel):
    st.subheader(ylabel)
    data = {}
    for dclass in dclasses:
        relevant_data = { x[0]:dclass['mutator'](x[1]) for x in list(dclass['data'].groupby(dclass['feature']))}
        data[ dclass['name'] ] = relevant_data
    df = pd.DataFrame(data)
    df.fillna(0, inplace= True)
    df = df.sort_index()

    fig1 = plt.figure(figsize=(20, 10))

    X = np.arange(len(df))

    cols = df.columns
    for i, c in enumerate(cols):
        plt.bar(X + i * .1, df[c], width = 0.1, label = c)

    plt.legend()
    plt.xlabel("Date (Month)")
    plt.xticks(X, list(df.index), rotation = 90)
    return st.pyplot(fig1)

    return st.line_chart(df)

# # Monthly transactions

cryptophunks_dataclass = {
    'name': 'CryptoPhunks',
    'data': cryptophunks_activities[ ['priceUsd', 'year-month']  ],
    'feature': 'year-month',
    'mutator': lambda x: len(x)
}


st.write("""

After investigating the suspiciously high trade volumes (especially during Jan 2022), there is evidence of potential wash trading.
The following section visualizes some of the suspicious activity that can potentially be considered as wash trading

""")

st.write("""

Top 10 transactions in CryptoPhunks

""")






















import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
wallets_to_idx = {}
idx_to_wallet = {}


def compress(wallet):
    if not wallet in wallets_to_idx:
        wallets_to_idx[wallet] = len(wallets_to_idx) + 1
        idx_to_wallet[ len(wallets_to_idx) ] = wallet



# Process transfers dataFrame
transactions_df = pd.read_json("./punks/ETHEREUM/0xf07468ead8cf26c752c676e43c814fee9c8cf402/activities_SELL.json")
transactions_df['tokenId'] = transactions_df.nft.apply(lambda x: x['type']['tokenId'])
transactions_df.rename(columns={
    'seller': 'from',
    'buyer': 'to',
}, inplace = True)

transactions_df['from'] = transactions_df['from'].apply(lambda x: x.split(":")[1])
transactions_df['to'] = transactions_df['to'].apply(lambda x: x.split(":")[1])

filtered_transactions_df = transactions_df[
    list(set(transactions_df.columns) - set([
        'cursor',
        'reverted',
        'sellerOrderHash',
#         'transactionHash',
        'amountUsd',
        'lastUpdatedAt',
        'buyerOrderHash',
        'payment',
        '@type',
        'nft',
        'blockchainInfo',
        'id',
    ]))
].copy()





common_cols = ['from', 'type', 'to', 'transactionHash', 'tokenId', 'date', 'price', 'priceUsd']

combined_df = filtered_transactions_df.copy()
combined_df.drop_duplicates(inplace=True)
combined_df.sort_values(by=['date'], ascending=False, inplace=True)

for i, row in combined_df.iterrows():
    compress(row['from'])
    compress(row['to'])

combined_df['from_compressed'] = combined_df['from'].apply(lambda x: wallets_to_idx[x])
combined_df['to_compressed'] = combined_df['to'].apply(lambda x: wallets_to_idx[x])

combined_df['year-month'] = combined_df['date'].apply(lambda x: '-'.join(str(x).split('-')[:2])) .copy()


combined_df.dropna(inplace = True)
combined_df = combined_df[ combined_df['priceUsd'] > 0 ]

st.write(combined_df.head(3))



sus_tokens = {}




import matplotlib.pyplot as plt
import numpy as np

def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=16,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items



def plotGraph(edge_list, node_size = 1000, node_font_color='white', edge_font_color='black', save_image = False, filename = 'plot.png', title = "Title", font_size=16, collection = 'NA'):
    G = nx.DiGraph()

    G.add_edges_from(edge_list)
    pos=nx.spring_layout(G,seed=5)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_title(title)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color=node_font_color)

    threshold = 5
    heavy_edges = []
    light_edges = []
        
    for edge in edge_list:
        if edge[2]['w'][1] >= threshold:
            heavy_edges.append(edge)
        else:
            light_edges.append(edge)
    
    
    def plotEdges(G, edgelist, color):
        if len(edgelist) == 0:
            return 
        edge_weights = [edge[2]['w'][1] for edge in edgelist]
        colors = [color]*len(edge_weights)
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        norm_weights = [(w - min_weight) / (max_weight - min_weight + 1) for w in edge_weights]
                
        
        curved_edges = []
        straight_edges = []
        
        straight_weights, curved_weights = [], []
        straight_alphas, curved_alphas = [], []
        
        C = 1.5
        for edge in edgelist:
            if (edge[1], edge[0]) not in G.edges:
                straight_edges.append( (edge[0], edge[1]) )
                straight_weights.append(edge[2]['w'][1])
                alpha = (edge[2]['w'][1] - min_weight + 1) / (max_weight - min_weight + 1)
                straight_alphas.append( alpha / C + (1.-1./C)  )
            else:
                curved_edges.append((edge[0], edge[1]))
                curved_weights.append(edge[2]['w'][1])
                alpha = (edge[2]['w'][1] - min_weight + 1) / (max_weight - min_weight + 1)
                curved_alphas.append( alpha / C + (1.-1./C) )  
                

        straight_edges_plot = nx.draw_networkx_edges(G, pos, edgelist = straight_edges, alpha = straight_alphas, arrows=True, edge_color=[color]*len(straight_edges), edge_cmap=plt.cm.Reds, width=2, edge_vmin=0, edge_vmax=1, arrowsize = 30)
        
        arc_rad = 0.25
        curved_edges_plot = nx.draw_networkx_edges(G, pos, edgelist = curved_edges, alpha = curved_alphas, arrows=True, edge_color=[color]*len(curved_edges), edge_cmap=plt.cm.Reds, connectionstyle=f'arc3, rad = {arc_rad}', width=2, edge_vmin=0, edge_vmax=1, arrowsize = 30)
        edge_weights = nx.get_edge_attributes(G,'w')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, font_color=edge_font_color, font_size=font_size)
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False, font_color=edge_font_color, font_size=font_size)
        
        
    plotEdges(G, heavy_edges, 'red')
    plotEdges(G, light_edges, 'green')


    img_path = f"./{collection}/{filename}"
    hf.makeFolderIfNotExists(f'./{collection}/')

    plt.savefig(img_path)
    if not collection in sus_tokens:
        sus_tokens[collection] = []
    sus_tokens[collection].append(img_path)

# edge_list_sample = [
#     ('Alice','Bob',{'w':(59.23, 1), 'color':'red', 'heavy': True}),
#     ('Bob','Alice',{'w':(60.88, 9), 'color':'red', 'heavy': True}),
#     ('Mark','Bob',{'w':(55.43, 4), 'color':'green', 'heavy': False}),
#     ('Alice','Mark',{'w':(60.19, 23), 'color':'green', 'heavy': False}),
# ]


# plotGraph(edge_list_sample, title = "Sample transactions network")


tmg = nx.MultiDiGraph()

for index, row in combined_df.iterrows():
    tmg.add_edges_from([
        (row['from'], row['to'], row['priceUsd'])
    ])

print(f"The transactions network contains {tmg.number_of_nodes()} unique nodes (wallets), and {tmg.number_of_edges()} unique transaction pairs") 


token_groups = list(combined_df.groupby('tokenId'))

def buildAndPlotGraph(token, dfs):
    
    for i in range(1, len(dfs)):
        dfs[0] = pd.concat([
            dfs[0], dfs[i]
        ])

    Gr = nx.MultiDiGraph()
    for i, row in dfs[0].iterrows():
        Gr.add_edges_from([
            (row['from_compressed'], row['to_compressed'], row['priceUsd'])
        ])
   
    return Gr.number_of_edges() >= Gr.number_of_nodes()



# Check wash trading

COLLECTION = 'cryptophunks'
def CheckWashTrading(df, threshold = 5, plot=True, collection = 'NA'):
    
    edge_list = []
    tokenId = list(df['tokenId'])[0]
    
    pair_groups = df.groupby(['from', 'to'])
    heavyCount = 0 
    for gr in pair_groups:
        u, v = wallets_to_idx[ gr[0][0] ], wallets_to_idx[ gr[0][1] ]
        w = len(gr[1])
        av =  round(gr[1].priceUsd.mean(), 1)
        edge_list.append(
            (u, v, {'w':(av,w), 'heavy': w>=threshold  })
        )
        heavyCount += (w>=threshold)
#     plotGraph(edge_list, title = "Sample transactions network", node_font_color='black', node_size=800)
    if plot:
        plotGraph(edge_list, title = f"Transactions for token: {tokenId} of collection ({COLLECTION})", node_size=1200, save_image = True, filename=f'{tokenId}.png', collection = collection)
    return heavyCount
    
# plotGraph(edge_list_sample, title = "Sample transactions network")




tokensWithCycles = []

for cand, (token, df) in enumerate((token_groups)):
    if buildAndPlotGraph(token, [df]):
        if CheckWashTrading(token_groups[ cand ][1], plot = False, collection = COLLECTION):
            
            CheckWashTrading(token_groups[ cand ][1], plot = True, collection = COLLECTION)



phunks_imgs = sus_tokens['cryptophunks']
st.header(f"There are {len(sus_tokens['cryptophunks'])} suspicious tokens in total:")

for img_path in phunks_imgs:
    tokenId = img_path.split('/')[-1].split('.')[0]
    tokenLink = hf.getLinkFromCollectionAndToken(COLLECTION_ADDRESSES['cryptophunks'], tokenId)
    st.markdown(f"[Token {tokenId}]({tokenLink})")

    lhs, rhs = st.columns(2)
    
    with lhs:
        st.image(img_path, width=700)
    
    with rhs:
        st.write(combined_df[ combined_df['tokenId'] == tokenId ][ ['from', 'to', 'price', 'year-month'] ].head(20) )
        st.image(f"./punks/ETHEREUM/{COLLECTION_ADDRESSES['cryptophunks']}/images/{tokenId}.png", width=250)
    

    st.markdown("---")









#RL
from tensorforce import Runner
from tensorforce import Environment
from tensorforce import Agent
#keras
import tensorflow as tf
#basic
import numpy as np
import pandas as pd
#sklearn
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
#image processing
from skimage.future import graph
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.segmentation import felzenszwalb
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.transform import rescale
#visialization
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_palette("Set2")
#skmultiflow
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.rules import VeryFastDecisionRulesClassifier
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
#scikit 
from scipy.spatial import distance_matrix
#graph
import networkx as nx
#ontology
import owlready2 as owl
#language model
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#other
from itertools import count

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape[0:2]
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2),
                    (0,0)
                    ),
                  mode = 'constant')

def visualize_objects(label_image,image):
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 5:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
def find_objects(image):
    img = rgb2gray(image)

    # apply threshold
    thresh = threshold_otsu(img)
    reg = closing(img > thresh, square(3))

    # label image regions
    label_image = label(reg)
    
    return label_image

def find_cut_objects(image,m_h,m_w):
    label_image=find_objects(image)
    regions=regionprops(label_image)
    i=len(regions)-1
    for reg in regions:
        minr, minc, maxr, maxc = reg.bbox
        split_h=np.linspace(0,(maxr-minr),1+int(np.ceil((maxr-minr)/m_h))).astype("int")
        split_w=np.linspace(0,(maxc-minc),1+int(np.ceil((maxc-minc)/m_w))).astype("int")
        for x in range(int(np.ceil((maxr-minr)/m_h))):
            for y in range(int(np.ceil((maxc-minc)/m_w))):
                i=i+1
                label_image[reg.slice][split_h[x]:split_h[x+1],split_w[y]:split_w[y+1]][reg.image[split_h[x]:split_h[x+1],split_w[y]:split_w[y+1]]]=i


    regions=regionprops(label_image)
    objects=np.array([to_shape(image[reg.slice],(m_h,m_w)) for reg in regions])
    return label_image, objects, regions


def regions_to_graph(regions,labels,prox=60):
    #weighted?
    centroids=np.array([reg.centroid for reg in regions])
    dist=distance_matrix(centroids,centroids)
    dist=(dist<prox)*dist
    G = nx.from_numpy_matrix(dist)
    lab={x:str(labels[x]) for x in range(len(centroids))}
    nx.set_node_attributes(G, lab, "feature")
    pos_x={x:centroids[x][0] for x in range(len(centroids))}
    nx.set_node_attributes(G, pos_x, "position_x")
    pos_y={x:centroids[x][1] for x in range(len(centroids))}
    nx.set_node_attributes(G, pos_y, "position_y")
    return G


def visualize_graph(g):
    groups = set(nx.get_node_attributes(g,'feature').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = g.nodes()
    colors = [mapping[g.nodes[n]['feature']] for n in nodes]  
    
    pos = nx.spring_layout(g)
    for n in nodes:
        pos[n][1]=-1*g.nodes[n]['position_x']
        pos[n][0]=g.nodes[n]['position_y']

    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()
    
def graph_to_embedding(Graphs,model,iterations=2,epochs=10,keep_train=True,rebuild_vocab=False,seed=42):
    Doc_all=[]
    for i in range(len(Graphs)):
        hashes=nx.weisfeiler_lehman_subgraph_hashes(Graphs[i], iterations=iterations, node_attr="feature")
        Doc=[]
        for node in Graphs[i].nodes:
            Doc=Doc+[Graphs[i].nodes[node]["feature"]]+hashes[node]
        
        Doc_all=Doc_all+[Doc.copy()]
    documents = [
            TaggedDocument(words=doc, tags=[str(i)])
            for i, doc in enumerate(Doc_all)
        ]
    
    if rebuild_vocab:
        if len(model.wv)>0:
            model.build_vocab(documents,update=True)
        else:
            model.build_vocab(documents)
        
    if keep_train:
        model.train(documents,total_examples=len(documents),epochs=epochs)
        
    #np.array([model.docvecs[str(i)] for i, _ in enumerate(Doc_all)])
    model.random.seed(seed)
    return np.array([model.infer_vector(doc) for doc in Doc_all])



class CustomEnvironment(Environment):

    def __init__(self, base_env):
        self.base_env=base_env
        self.num_clusters_obj=16
        self.image_seg=Birch(n_clusters=None,threshold=128,branching_factor=100)
        self.pca_obj=IncrementalPCA(n_components=50)
        self.max_obj_width=160//10
        self.max_obj_height=210//10
        self.prox=40
        self.state_param={}
        self.state_param["type"]="float"
        self.state_param["shape"]=128
        self.state_param["min_value"]=-1
        self.state_param["max_value"]=1
        self.example_objects={}
        self.example_objects_pca={}
        self.total_timestep=0
        self.keep_train=True
        self.rebuild_vocab=True
        self.iterations=1
        #graph list 
        self.graph_list=[]
        #semantic model
        self.dimensions=128
        self.min_count=1
        self.seed=42
        self.workers = 4
        self.epochs = 20
        self.learning_rate = 0.0025
        self.window=5
        self.hs=1
        self.dm=1
        self.negative=0
        self.semantic_model=Doc2Vec(
            vector_size=self.dimensions,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,
            hs=self.hs,
            negative=self.negative,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed
        )
        
        super().__init__()

    def states(self):
        return self.state_param

    def actions(self):
        return self.base_env.actions()

    def train_semantic(self):
        #make embedding
        graph_to_embedding(self.graph_list,self.semantic_model,iterations=self.iterations,epochs=self.epochs,keep_train=True,rebuild_vocab=True,seed=42)
        
    
    def preprocess_state(self, state, train=False, rebuild_vocab=False, train_semantic=False):
        
        #cut objects
        label_image, objects, regions=find_cut_objects(state,self.max_obj_height,self.max_obj_width)
        X=objects.reshape(objects.shape[0],-1)
        
        #PCA
        if train:
            self.pca_obj.partial_fit(np.vstack((X,X)))
            
        X=self.pca_obj.transform(X)
        
        #BIRCH clustering
        if train:
            self.image_seg.partial_fit(X)
            if self.image_seg.n_clusters is None and len(self.image_seg.subcluster_labels_)>self.num_clusters_obj-1:
                self.image_seg.set_params(n_clusters=self.num_clusters_obj-1)
                self.image_seg.partial_fit()
                
        labels=self.image_seg.predict(X)
            
        for i in range(len(labels)):
            if train:
                self.example_objects[labels[i]]=state[regions[i].slice]
                self.example_objects_pca[labels[i]]=self.pca_obj.inverse_transform(X[i]).reshape(self.max_obj_height,self.max_obj_width,3)
            
            label_image[regions[i].slice][regions[i].image]=labels[i]+1
        
        #make graph
        graph=regions_to_graph(regions,labels,prox=self.prox)
        
        #make embedding
        embedding=graph_to_embedding([graph],self.semantic_model,iterations=self.iterations,epochs=self.epochs,keep_train=train_semantic,rebuild_vocab=rebuild_vocab,seed=42)
        
        #clip emmbeding
        embedding=np.clip(embedding,self.state_param["min_value"], self.state_param["max_value"])[0]
        #smaller image
        label_image=np.max(np.dstack((label_image[0::2,0::2],label_image[1::2,1::2])),axis=-1)
        return label_image.astype("int8"),regions,labels,graph,embedding
        
        
        
    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; restrict training timesteps via
    #     Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = self.base_env.reset()
        _,_,_,graph,state = self.preprocess_state(state, train=True, rebuild_vocab=True, train_semantic=True)
        return state

    def execute(self, actions):
        next_state, terminal, reward = self.base_env.execute(actions)
        
        self.total_timestep+=1
        if self.total_timestep<500:
            self.keep_train=True
        else:
            self.keep_train=False
        
        if self.total_timestep%20==0:
            self.train_semantic()
            self.graph_list=[]
            print(self.total_timestep)
        
        _,_,_,graph,next_state=self.preprocess_state(next_state,self.keep_train)
        self.graph_list.append(graph)
        
        return next_state, terminal, reward
    

    
trace=np.load("./record/trace-000000000.npz")

States=trace["states"][0:3]

Rewards=trace["reward"]

Actions=trace["actions"]

# OpenAI-Gym environment specification
environment = Environment.create(
       environment='gym', level='SpaceInvaders-v4',max_episode_timesteps=1000)

custom=CustomEnvironment(environment)

custom.preprocess_state(States[0], train=True, rebuild_vocab=True, train_semantic=True)

Graphs=[]
for state in States:
    _,regions,labels,_,_=custom.preprocess_state(state, train=False)
    Graphs.append(regions_to_graph(regions,labels,prox=40))
    
X=graph_to_embedding(Graphs[0:2],custom.semantic_model,iterations=1,epochs=2,keep_train=True,rebuild_vocab=True)

print("hello")

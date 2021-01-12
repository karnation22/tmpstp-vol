## NOT KARN'S WORK!
# -*- coding: utf-8 -*-
import os
import pandas as pd
import argparse
import re
import spacy
from spacy.attrs import intify_attrs
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

import neuralcoref

import networkx as nx
import matplotlib.pyplot as plt

#nltk.download('stopwords')
from nltk.corpus import stopwords
all_stop_words = ['many', 'us', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                  'today', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                  'september', 'october', 'november', 'december', 'today', 'old', 'new']
all_stop_words = sorted(list(set(all_stop_words + list(stopwords.words('english')))))

abspath = os.path.abspath('') ## String which contains absolute path to the script file
#print(abspath)
os.chdir(abspath)

### ==================================================================================================
# Tagger

def get_tags_spacy(nlp, text):
    doc = nlp(text)    
    entities_spacy = [] # Entities that Spacy NER found
    for ent in doc.ents:
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    return entities_spacy

def tag_all(nlp, text, entities_spacy):
    ## print("here3_1")
    if ('neuralcoref' in nlp.pipe_names):
        nlp.pipeline.remove('neuralcoref')  
    ## print("here3_2")  
    neuralcoref.add_to_pipe(nlp) # Add neural coref to SpaCy's pipe 
    ## print("here3_3")   
    ## print("text3_3_2", text)
    doc = nlp(text)
    ## print("here3_4")
    return doc

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

def tag_chunks(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))

def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))

def clean(text):
    ## print("text: ",text)
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    ## print("here1")
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    ## print("here2")
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    ## print("here3")
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    ## print("here4")
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    ## print("here5")
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    ## print("here6")
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    ## print("here7")
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    ## print("here8")
    
    text = re.sub('’', "'", text, flags=re.UNICODE)           # curly apostrophe
    ## print("here9")
    text = re.sub('‘', "'", text, flags=re.UNICODE)           # curly apostrophe
    ## print("here10")
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    ## print("here11")
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    ## print("here12")
    text = re.sub("\|", ", ", text, flags=re.UNICODE)
    ## print("here13")
    text = text.replace('\t', ' ')
    
    text = re.sub('…', '.', text, flags=re.UNICODE)           # elipsis
    ## print("here14")
    text = re.sub('â€¦', '.', text, flags=re.UNICODE)      
    ## print("here15")    
    text = re.sub('â€“', '-', text)           # long hyphen
    ## print("here16")
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    ## print("here17")
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()
    ## print("here18")
    ## print("text: ",text)
    return text

def tagger(text):  
    df_out = pd.DataFrame(columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency'])
    corefs = []
    text = clean(text)
    ## print("here2_1")
    nlp = spacy.load("en_core_web_sm")
    ## print("here2_2")
    entities_spacy = get_tags_spacy(nlp, text)
    ## print("here2_3")
    ## print("SPACY entities:\n", ([ent for ent in entities_spacy]), '\n\n')
    document = tag_all(nlp, text, entities_spacy)
    ## print("here2_4")
    print(document)
    #for token in document:
    #    print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])
    
    ### Coreferences
    if document._.has_coref:
        for cluster in document._.coref_clusters:
            main = cluster.main
            for m in cluster.mentions:                    
                if (str(m).strip() == str(main).strip()):
                    continue
                corefs.append([str(m), str(main)])
    tag_chunks(document)    
    
    # chunk - somethin OF something
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (w_middle.text == 'of'): # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change, 'ENTITY')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part 
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
                #print(span)
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, 'ENTITY')
            
    # chunk entities
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')
    
    doc_id = 1
    count_sentences = 0
    prev_dep = 'nsubj'
    for token in document:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, prev_dep]
        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
                  
        if (token.text == '.'):
            count_sentences += 1
        prev_dep = token.dep_
        
    return df_out, corefs

### ==================================================================================================
### triple extractor

def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        if spo == 'predicate' and w != "'s" and w != "\"": #= 11.95
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    return predicates

def get_subjects(s, start, end, adps):
    subjects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'subject' in spo or 'entity' in spo or 'object' in spo:
                subjects[index] = w
    return subjects
    
def get_objects(s, start, end, adps):
    objects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'object' in spo or 'entity' in spo or 'subject' in spo:
                objects[index] = w
    return objects

def get_positions(s, start, end):
    adps = {}
    for w, index, spo in s:        
        if index >= start and index <= end:
            if 'of' == spo or 'at' == spo:
                adps[index] = w
    return adps

def create_triples(df_text, corefs):
    sentences = []
    aSentence = []
    
    for index, row in df_text.iterrows():
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep = row.items()
        if 'subj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'subject'])
        elif 'ROOT' in dep[1] or 'VERB' in cg_pos[1] or pos[1] == 'IN':
            aSentence.append([word[1], word_id[1], 'predicate'])
        elif 'obj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'object'])
        elif ent[1] == 'ENTITY':
            aSentence.append([word[1], word_id[1], 'entity'])        
        elif word[1] == '.':
            sentences.append(aSentence)
            aSentence = []
        else:
            aSentence.append([word[1], word_id[1], pos[1]])
    
    relations = []
    #loose_entities = []
    for s in sentences:
        if len(s) == 0: continue
        preds = get_predicate(s) # Get all verbs
        """
        if preds == {}: 
            preds = {p[1]:p[0] for p in s if (p[2] == 'JJ' or p[2] == 'IN' or p[2] == 'CC' or
                     p[2] == 'RP' or p[2] == ':' or p[2] == 'predicate' or
                     p[2] =='-LRB-' or p[2] =='-RRB-') }
            if preds == {}:
                #print('\npred = 0', s)
                preds = {p[1]:p[0] for p in s if (p[2] == ',')}
                if preds == {}:
                    ents = [e[0] for e in s if e[2] == 'entity']
                    if (ents):
                        loose_entities = ents # not significant for now
                        #print("Loose entities = ", ents)
        """
        if preds:
            if (len(preds) == 1):
                #print("preds = ", preds)
                predicate = list(preds.values())[0]
                if (len(predicate) < 2):
                    predicate = 'is'
                #print(s)
                ents = [e[0] for e in s if e[2] == 'entity']
                #print('ents = ', ents)
                for i in range(1, len(ents)):
                    relations.append([ents[0], predicate, ents[i]])

            pred_ids = list(preds.keys())
            pred_ids.append(s[0][1])
            pred_ids.append(s[len(s)-1][1])
            pred_ids.sort()
                    
            for i in range(1, len(pred_ids)-1):
                predicate = preds[pred_ids[i]]
                adps_subjs = get_positions(s, pred_ids[i-1], pred_ids[i])
                subjs = get_subjects(s, pred_ids[i-1], pred_ids[i], adps_subjs)
                adps_objs = get_positions(s, pred_ids[i], pred_ids[i+1])
                objs = get_objects(s, pred_ids[i], pred_ids[i+1], adps_objs)
                for k_s, subj in subjs.items():                
                    for k_o, obj in objs.items():
                        obj_prev_id = int(k_o) - 1
                        if obj_prev_id in adps_objs: # at, in, of
                            relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj])
                        else:
                            relations.append([subj, predicate, obj])
    
    ### Read coreferences: coreference files are TAB separated values
    coreferences = []
    for val in corefs:
        if val[0].strip() != val[1].strip():
            if len(val[0]) <= 50 and len(val[1]) <= 50:
                co_word = val[0]
                real_word = val[1].strip('[,- \'\n]*')
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
            else:
                co_word = val[0]
                real_word = ' '.join((val[1].strip('[,- \'\n]*')).split()[:7])
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
                
    # Resolve corefs
    triples_object_coref_resolved = []
    triples_all_coref_resolved = []
    for s, p, o in relations:
        coref_resolved = False
        for co in coreferences:
            if (s == co[0]):
                subj = co[1]
                triples_object_coref_resolved.append([subj, p, o])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_object_coref_resolved.append([s, p, o])

    for s, p, o in triples_object_coref_resolved:
        coref_resolved = False
        for co in coreferences:
            if (o == co[0]):
                obj = co[1]
                triples_all_coref_resolved.append([s, p, obj])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_all_coref_resolved.append([s, p, o])
    return(triples_all_coref_resolved)

### ==================================================================================================
## Get more using Network shortest_paths

def get_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, key=p)
    return G

def get_entities_with_capitals(G):
    entities = []
    for node in G.nodes():
        if (any(ch.isupper() for ch in list(node))):
            entities.append(node)
    return entities

def get_paths_between_capitalised_entities(triples):
    
    g = get_graph(triples)
    ents_capitals = get_entities_with_capitals(g)
    paths = []
    #print('\nShortest paths among capitalised words -------------------')
    for i in range(0, len(ents_capitals)):
        n1 = ents_capitals[i]
        for j in range(1, len(ents_capitals)):
            try:
                n2 = ents_capitals[j]
                path = nx.shortest_path(g, source=n1, target=n2)
                if path and len(path) > 2:
                    paths.append(path)
                path = nx.shortest_path(g, source=n2, target=n1)
                if path and len(path) > 2:
                    paths.append(path)
            except Exception:
                continue
    return g, paths

def get_paths(doc_triples):
    triples = []
    g, paths = get_paths_between_capitalised_entities(doc_triples)
    for p in paths:
        path = [(u, g[u][v]['key'], v) for (u, v) in zip(p[0:], p[1:])]
        length = len(p)
        if (path[length-2][1] == 'in' or path[length-2][1] == 'at' or path[length-2][1] == 'on'):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], path[length-2][1], path[length-2][2]])
        elif (' in' in path[length-2][1] or ' at' in path[length-2][1] or ' on' in path[length-2][1]):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], 'in', path[length-2][2]])
    for t in doc_triples:
        if t not in triples:
            triples.append(t)
    return triples

def get_center(nodes):
    center = ''
    if (len(nodes) == 1):
        center = nodes[0]
    else:   
        # Capital letters and longer is preferred
        cap_ents = [e for e in nodes if any(x.isupper() for x in e)]
        if (cap_ents):
            center = max(cap_ents, key=len)
        else:
            center = max(nodes, key=len)
    return center

def connect_graphs(mytriples):
    G = nx.DiGraph()
    for s, p, o in mytriples:
        G.add_edge(s, o, p=p)        
    
    """
    # Get components
    graphs = list(nx.connected_component_subgraphs(G.to_undirected()))
    
    # Get the largest component
    largest_g = max(graphs, key=len)
    largest_graph_center = ''
    largest_graph_center = get_center(nx.center(largest_g))
    
    # for each graph, find the centre node
    smaller_graph_centers = []
    for g in graphs:        
        center = get_center(nx.center(g))
        smaller_graph_centers.append(center)

    for n in smaller_graph_centers:
        if (largest_graph_center is not n):
            G.add_edge(largest_graph_center, n, p='with')
    """
    return G
        
def rank_by_degree(mytriples): #, limit):
    G = connect_graphs(mytriples)
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    
    # Use this to draw the graph
    draw_graph_centrality(G, degree_dict)

    Egos = nx.DiGraph()
    for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['degree'], reverse=True):
        ego = nx.ego_graph(G, a)
        Egos.add_edges_from(ego.edges(data=True))
        Egos.add_nodes_from(ego.nodes(data=True))
        
        #if (nx.number_of_edges(Egos) > 20):
        #    break
       
    ranked_triples = []
    for u, v, d in Egos.edges(data=True):
        ranked_triples.append([u, d['p'], v])
    return ranked_triples
    
def extract_triples(text):
    df_tagged, corefs = tagger(text)
    doc_triples = create_triples(df_tagged, corefs)
    all_triples = get_paths(doc_triples)
    filtered_triples = []    
    for s, p, o in all_triples:
        if ([s, p, o] not in filtered_triples):
            if s.lower() in all_stop_words or o.lower() in all_stop_words:
                continue
            elif s == p:
                continue
            if s.isdigit() or o.isdigit():
                continue
            if '%' in o or '%' in s: #= 11.96
                continue
            if (len(s) < 2) or (len(o) < 2):
                continue
            if (s.islower() and len(s) < 4) or (o.islower() and len(o) < 4):
                continue
            if s == o:
                continue            
            subj = s.strip('[,- :\'\"\n]*')
            pred = p.strip('[- :\'\"\n]*.')
            obj = o.strip('[,- :\'\"\n]*')
            
            for sw in ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who', 'that', 'this', 'these', 'those']:
                subj = ' '.join(word for word in subj.split() if not word == sw)
                obj = ' '.join(word for word in obj.split()  if not word == sw)
            subj = re.sub("\s\s+", " ", subj)
            obj = re.sub("\s\s+", " ", obj)
            
            if subj and pred and obj:
                filtered_triples.append([subj, pred, obj])

    TRIPLES = rank_by_degree(filtered_triples)
    ## TRIPLES_ch = [ [translator.en2ch(source) for source in triple] for triple in TRIPLES]

    return TRIPLES

def draw_graph_centrality_helper(dictionary):
    dictVol, dictStress = dict(), dict()
    for dirItem in os.listdir():
        if("voice_stress_and_vol" in dirItem): 
            os.chdir(dirItem)
            break
    ## process voice stress first
    for item in dictionary:
        dictStress[item] = []
        for word in item.split():
            for subDirItem in os.listdir():
                if(word in subDirItem): ##if word in chunk in an rms file..
                    with open(subDirItem,"r") as f_subDir:
                        strSubDir = f_subDir.read().split()
                        relVal = float(strSubDir[1])
                    dictStress[item].append(relVal)
                    break
        try: dictStress[item] = sum(dictStress[item])/len(dictStress[item])
        except: dictStress[item] = 0.5
    ## next process volume
    ## print(os.getcwd())
    for subDirItem in os.listdir():
        ## print("subDirItem: ",subDirItem)
        if("mono_RMS_vol" in subDirItem):
            stemFile = subDirItem[:subDirItem.index("_")]
            with open(subDirItem, "r") as f_sub_dir:
                allItems = f_sub_dir.readlines()
            break
    for item in dictionary:
        dictVol[item] = []
        for word in item.split():
            for volItem in allItems:
                if(word in volItem):
                    relStr = volItem.strip()
                    relVal = float(relStr.split()[1])
                    dictVol[item].append(relVal)
                    break
        try: dictVol[item] = sum(dictVol[item])/len(dictVol[item])
        except: dictVol[item] = 0.5
            
    return dictVol, dictStress, stemFile

                    
def draw_graph_centrality(G, dictionary):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G)
    ## print("Nodes\n", G.nodes(True))
    ## print("Edges\n", G.edges())
    ## print(G.nodes().data())
    ## print("dictionary: ",dictionary)
    dictVol, dictStress, stemFile = draw_graph_centrality_helper(dictionary)
    ##dictVol, dictStress have same keys as dictionary; 
    #   values are 0.0 to 1.0 floating point of volume//voice stress..

    nx.draw_networkx_nodes(G, pos, 
            nodelist=dictionary.keys(),
            with_labels=False,
            edge_color='black',
            width=1,
            linewidths=1,
            ## node_size = [v * 150 for v in dictionary.values()], ##minor change here... 
            node_size= [v * 300 for v in dictVol.values()],
            ## node_color = 'blue', ##minor change here....
            node_color= [ i for i in dictStress.values()],
            alpha=0.5)
    edge_labels = {(u, v): d["p"] for u, v, d in G.edges(data=True)}
    #print(edge_labels)
    nx.draw_networkx_edge_labels(G, pos,
                           font_size=10,
                           edge_labels=edge_labels,
                           font_color='blue')
    nx.draw(G, pos, with_labels=True, node_size=1, node_color='blue')
    os.chdir("..")
    networkXImageFile = "{}networkXGraph.png".format(stemFile)
    plt.savefig(networkXImageFile)
    if("{}_misc_data".format(stemFile) not in os.listdir()):
        os.mkdir("{}_misc_data".format(stemFile))
    os.system("mv {}textString.txt {}networkXGraph.png {}_misc_data"
            .format(stemFile,stemFile,stemFile))
    
if __name__ == "__main__":
    # """
    # Celebrity chef Jamie Oliver's British restaurant chain has become insolvent, putting 1,300 jobs at risk. The firm said Tuesday that it had gone into administration, a form of bankruptcy protection, and appointed KPMG to oversee the process.The company operates 23 Jamie's Italian restaurants in the U.K. The company had been seeking buyers amid increased competition from casual dining rivals, according to The Guardian. Oliver began his restaurant empire in 2002 when he opened Fifteen in London. Oliver, known around the world for his cookbooks and television shows, said he was "deeply saddened by this outcome and would like to thank all of the staff and our suppliers who have put their hearts and souls into this business for over a decade. "He said "I appreciate how difficult this is for everyone affected." I’m devastated that our much-loved UK restaurants have gone into administration.
    # """
    # """BYD debuted its E-SEED GT concept car and Song Pro SUV alongside its all-new e-series models at the Shanghai International Automobile Industry Exhibition. The company also showcased its latest Dynasty series of vehicles, which were recently unveiled at the company’s spring product launch in Beijing."""
    parser = argparse.ArgumentParser(description="Please input a text file")
    parser.add_argument("--text_file",help="text file to be parsed..", required=False, default=None)
    args = vars(parser.parse_args())
    if(args['text_file']==None or args['text_file'] not in os.listdir() or not args['text_file'].endswith(".txt")):
        text = """
        BYD debuted its E-SEED GT concept car and Song Pro SUV alongside its all-new e-series models at the Shanghai International Automobile Industry Exhibition. The company also showcased its latest Dynasty series of vehicles, which were recently unveiled at the company’s spring product launch in Beijing. A total of 23 new car models were exhibited at the event, held at Shanghai’s National Convention and Exhibition Center, fully demonstrating the BYD New Architecture (BNA) design, the 3rd generation of Dual Mode technology, plus the e-platform framework. Today, China’s new energy vehicles have entered the ‘fast lane’, ushering in an even larger market outbreak. Presently, we stand at the intersection of old and new kinetic energy conversion for mobility, but also a new starting point for high-quality development. To meet the arrival of complete electrification, BYD has formulated a series of strategies, and is well prepared.
        """
    else:
        with open(args['text_file'],'r') as f_args:
            text = f_args.read()
    # """
    # An arson fire caused an estimated $50,000 damage at a house on Mt. Soledad that was being renovated, authorities said Friday.San Diego police were looking for the arsonist, described as a Latino man who was wearing a red hat, blue shirt and brown pants, and may have driven away in a small, black four-door car.A resident on Palomino Court, off Soledad Mountain Road, called 9-1-1 about 9:45 a.m. to report the house next door on fire, with black smoke coming out of the roof, police said. Firefighters had the flames knocked down 20 minutes later, holding the damage to the attic and roof, said City spokesperson Alec Phillip. No one was injured.Metro Arson Strike Team investigators were called and they determined the blaze had been set intentionally, Phillip said.Police said one or more witnesses saw the suspect run south from the house and possibly leave in the black car.
    # """
    mytriples = extract_triples(text)
    print('\n\nFINAL TRIPLES = ', len(mytriples))
    for t in mytriples:
        print(t)
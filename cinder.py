import pickle
import os
import numpy as np
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import docx
from sklearn.linear_model import LinearRegression
import spacy
import spacy.symbols as symbols
from scipy.sparse import dok_matrix, csr_matrix

nlp = spacy.load('en_core_web_lg')


def convert_pdf(fname):
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, set()):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text


def convert_docx(fname):
    doc = docx.Document(fname)
    text = []
    for para in doc.paragraphs:
        txt = para.text.encode('ascii', 'ignore')
        text.append(txt.decode('ascii'))
    text = '\n'.join(text)
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text


def convert_all_cvs(folder):
    all_cvs = {}
    for fname in os.listdir(folder):
        print(fname)
        if fname.endswith('.pdf'):
            all_cvs[fname[:-4]] = convert_pdf(os.path.join(folder, fname))
        elif fname.endswith('.docx'):
            all_cvs[fname[:-5]] = convert_docx(os.path.join(folder, fname))
        elif '.' in fname:
            print('Unknown file type:', fname.split('.')[-1])
    pickle.dump(all_cvs, open(os.path.join(folder, 'all_cvs.pkl'), 'wb'))
    return all_cvs


def create_words_by_cv(all_cvs, folder):

    ent_escapes = {'PERSON', 'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}
    pos_escapes = {'NUM', 'CCONJ', 'PRON', 'SYM', 'PART', 'DET', 'ADP', 'ADV', 'AUX', 'CONJ'}
    dep_escapes = {'punct', 'aux', 'auxpass', 'poss', 'advmod', 'advcl', 'ccomp'}

    words_by_cv = {}
    for name, text in all_cvs.items():
        print(name)
        doc = nlp(text)
        words = set()
        for token in doc:
            if token.is_space or token.is_stop or token.is_digit or token.is_punct or token.is_bracket or token.like_url or token.like_email:
                continue
            if token.ent_type_ in ent_escapes or token.pos_ in pos_escapes or token.dep_ in dep_escapes:
                continue
            word = token.text
            word = ''.join([c for c in word if c.isalpha()])
            if len(word) == 0:
                continue
            txt = nlp(word)[0].lemma_
            # txt = token.text if token.ent_type_ != '' else token.lemma_
            words.add(txt)
        words_by_cv[name] = words
    pickle.dump(words_by_cv, open(os.path.join(resume_folder, 'words_by_cv.pkl'), 'wb'))
    return words_by_cv


class Vocabulary:

    def __init__(self, words=[], similarity_cutoff=0.5):
        self.words = []
        self.indices = {}
        self.similarities = []
        self.cutoff = similarity_cutoff
        self.tokens = []
        ls = len(words)
        for iword in range(len(words)):
            word = words[iword]
            self.add_word(word)
            print(str((iword + 1) / ls * 100) + '%:', word)

    def add_word(self, word):
        if word in self.indices:
            return True
        token = nlp(word)[0]
        if len(token.text) <= 1:
            print('WARNING: cannot add', word)
            return False
        idx = len(self.words)
        self.indices[word] = idx
        self.tokens.append(token)
        self.words.append(word)
        similarities = np.array([token.similarity(t) for t in self.tokens])
        isims = np.argwhere(similarities >= self.cutoff).ravel()
        sims = similarities[isims]
        self.similarities.append(np.vstack([isims, sims]))
        for ii in range(len(isims)):
            i = isims[ii]
            if i != idx:
                self.similarities[i] = np.append(self.similarities[i], np.array([idx, sims[ii]]).reshape(-1, 1), axis=1)
        return True

    def save(self):
        return {
            'words':        self.words,
            'indices':      self.indices,
            'similarities': self.similarities,
            'cutoff':       self.cutoff
        }

    def load(self, data):
        self.words = data['words']
        self.indices = data['indices']
        self.similarities = data['similarities']
        self.cutoff = data['cutoff']
        doc = nlp(' '.join(self.words))
        self.tokens = [token for token in doc]

    def get_similarities(self, target_words, text, cutoff):
        target_words = [word for word in target_words if len(word) > 1]
        exceptions = []
        for target_word in target_words:
            if target_word not in self.indices:
                if not self.add_word(target_word):
                    exceptions.append(target_word)
        if len(exceptions) > 0:
            target_words = [word for word in target_words if word not in exceptions]

        exceptions = []
        text = [t for t in text if len(t) > 1]
        for word in text:
            if word not in self.indices:
                if not self.add_word(word):
                    exceptions.append(word)
        if len(exceptions) > 0:
            text = [word for word in text if word not in exceptions]

        target_indices = [self.indices[target_word] for target_word in target_words]
        text_indices = np.array([self.indices[word] for word in text])
        results = np.zeros((len(target_words), cutoff))
        for i in range(len(target_words)):
            target_idx = target_indices[i]
            sims = self.similarities[target_idx]
            hits = np.intersect1d(text_indices, sims[0])
            indices = np.searchsorted(sims[0], hits)
            values = sims[1, indices]
            if len(values) > cutoff:
                values = np.sort(values)[::-1][:cutoff]
            results[i][:len(values)] = values
        return results


def ngram(words_by_cv):
    all_words = set()
    for name, words in words_by_cv.items():
        all_words = all_words.union(words)
    all_words = list(all_words)
    all_words.sort()
    all_words = [word for word in all_words if len(word) > 1]
    word_mapping = dict(zip(all_words, np.arange(len(all_words))))
    single_counts = np.zeros((len(all_words)))
    pair_counts = np.zeros((len(all_words), len(all_words)))
    for name, words in words_by_cv.items():
        word_ids = np.array([word_mapping[w] for w in words if w in word_mapping]).astype(np.int)
        single_counts[word_ids] += 1
        pair_counts[np.tile(word_ids, len(word_ids)), np.repeat(word_ids, len(word_ids))] += 1
    diag = np.diag(pair_counts) - 1
    for i in range(len(all_words)):
        pair_counts[i, i] = 0
    pair_counts = np.where(pair_counts == 0, 0, pair_counts - 1)

    s = np.sum(pair_counts, axis=0)
    nonzero_rows = np.nonzero(s)[0].astype(np.int)
    s = s[nonzero_rows]
    all_words = [all_words[ir] for ir in nonzero_rows]
    word_mapping = dict(zip(all_words, np.arange(len(all_words))))
    pair_counts = pair_counts[nonzero_rows][:, nonzero_rows]
    single_counts = single_counts[nonzero_rows]

    diag = diag[nonzero_rows]
    for i in range(diag.shape[0]):
        pair_counts[i, i] = diag[i]

    pair_counts = pair_counts / s
    single_counts = single_counts / np.sum(single_counts)

    cond_probs = pair_counts.T.copy()
    cond_probs = np.where(cond_probs == 0, 1.00, cond_probs / single_counts)
    cond_probs = cond_probs.T
    return word_mapping, np.log(cond_probs)


def rank_cvs2(words_by_cv, keywords):
    word_mapping, llhds = ngram(words_by_cv)
    all_words = set()
    for name, words in words_by_cv.items():
        all_words = all_words.union(words)
    all_words = list(all_words)
    all_words.sort()

    print(len(all_words), np.sum([len(words_by_cv[name]) for name in words_by_cv]))

    word_map = dict(zip(list(all_words), np.arange(len(all_words)).astype(np.int)))
    word_count = np.zeros(len(all_words))
    for name, words in words_by_cv.items():
        if len(words) == 0:
            continue
        # print(name)
        indices = [word_map[word] for word in words]
        n = len(words) - 1
        word_count[indices] += n - 1

    sort_indices = np.argsort(word_count)[::-1].astype(np.int)
    sorted_words = [all_words[i] for i in sort_indices]
    sorted_counts = word_count[sort_indices]

    all_counts = dict(zip(sorted_words, sorted_counts))

    new_keywords = {}
    for k in keywords:
        if k in word_mapping:
            new_keywords[k] = keywords[k]
    keywords = new_keywords

    kdoc = nlp(' '.join(list(keywords.keys())))
    corrs = np.ones((len(keywords), len(keywords)))
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            print(kdoc[j].text, kdoc[i].text)
            corrs[i, j] = corrs[j, i] = kdoc[i].similarity(kdoc[j])
    icorrs = np.linalg.inv(corrs)
    ic = np.linalg.cholesky(icorrs)

    scores = {}
    cutoff = 5
    names = list(words_by_cv.keys())
    for name in names:
        words = words_by_cv[name]
        if len(words) == 0:
            continue
        print(name)
        doc = nlp(' '.join(words))
        score = []
        all_tokens = [t2 for t2 in doc if t2.text in all_counts and t2.text in word_mapping]
        all_ids = [word_mapping[t.text] for t in all_tokens]
        for t in kdoc:
            if t.text not in word_mapping:
                continue
            it = word_mapping[t.text]
            score.append(np.sum(llhds[all_ids, it]) * keywords[t.text])
        score = np.array(score)
        # score = np.dot(score.reshape(1, -1), ic)
        score = np.sqrt(np.sum(score ** 2))
        scores[name] = score / len(all_tokens)

    sort_indices = np.argsort(list(scores.values()))[::-1].astype(np.int)
    names = list(scores.keys())
    sorted_names = [names[i] for i in sort_indices]
    return scores, sorted_names


def rank_cvs(words_by_cv, keywords, weights):
    all_words = set()
    for name, words in words_by_cv.items():
        all_words = all_words.union(words)
    all_words = list(all_words)
    all_words.sort()

    print(len(all_words), np.sum([len(words_by_cv[name]) for name in words_by_cv]))

    word_map = dict(zip(list(all_words), np.arange(len(all_words)).astype(np.int)))
    word_count = np.zeros(len(all_words))
    for name, words in words_by_cv.items():
        if name not in weights:
            continue
        if len(words) == 0:
            continue
        # print(name)
        indices = [word_map[word] for word in words]
        n = len(words) - 1
        word_count[indices] += n - 1

    sort_indices = np.argsort(word_count)[::-1].astype(np.int)
    sorted_words = [all_words[i] for i in sort_indices]
    sorted_counts = word_count[sort_indices]
    # for i in range(len(all_words)):
    #     print(sorted_words[i], sorted_counts[i])

    all_counts = dict(zip(sorted_words, sorted_counts))

    kdoc = nlp(' '.join(list(keywords.keys())))
    corrs = np.ones((len(keywords), len(keywords)))
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            print(kdoc[j].text, kdoc[i].text)
            corrs[i, j] = corrs[j, i] = kdoc[i].similarity(kdoc[j])
    icorrs = np.linalg.inv(corrs)
    ic = np.linalg.cholesky(icorrs)

    # pca = PCA()
    # pca

    scores = {}
    cutoff = 5
    names = list(words_by_cv.keys())
    for name in names:
        if name not in weights:
            continue
        words = words_by_cv[name]
        if len(words) == 0:
            continue
        print(name)
        length = len(words)
        doc = nlp(' '.join(words))
        score = []
        all_tokens = [t2 for t2 in doc if t2.text in all_counts and len(t2.text) > 1]
        counts = np.array([all_counts[t2.text] for t2 in all_tokens])
        for t in kdoc:
            similarities = []
            # for t2 in all_tokens:
            #     print(t2.text)
            #     similarities.append(t.similarity(t2))
            similarities = np.array([t.similarity(t2) for t2 in all_tokens])
            sim_indices = np.argsort(similarities)[::-1][:cutoff].astype(np.int)
            score.append(np.sum(similarities[sim_indices]) * keywords[t.text]) # * counts[sim_indices])
            #
            # count = all_counts[token.text]
            # similarities = np.array([token.similarity(t) for t in kdoc])
            # score += np.sum(np.exp(-(1.0 - similarities) * 10)) * count
        # score /= length
        score = np.array(score)
        score = np.dot(score.reshape(1, -1), ic)
        score = np.sum(score ** 2)
        scores[name] = np.sqrt(score) * weights[name]
        # print(name, score)

    sort_indices = np.argsort(list(scores.values()))[::-1].astype(np.int)
    names = list(scores.keys())
    sorted_names = [names[i] for i in sort_indices]
    return scores, sorted_names


def generate_keywords(category):
    skills = dict()
    skills['backend_programming'] = ['python', 'numpy', 'pandas', 'linux', 'ubuntu', 'unix']  # contains
    skills['optional_backend'] = ['c++', '.net', 'java']  # java no script
    skills['framework'] = ['django', 'flask', 'rest framework', 'docker']
    skills['database'] = ['sql', 'postgresql', 'orm', 'mongo', 'mysql', 'database', 'data mining', 'mongodb']
    skills['devops'] = ['aws', 'terraform', 'shell scripting', 'shell script', 'bash', 'memory management', 'backup', 'migration', 'recovery', 'cloning', 'big query', 'cloud']
    skills['quant_programming'] = ['tensor', 'keras', 'theano', 'tensorflow', 'pytorch', 'xgboost', 'lightgbm', 'sklearn', 'scikit', 'scipy']
    skills['mathematical_finance'] = ['black scholes', 'stochastic process', 'stochastic equation', 'martingale', 'geometric brownian motion', 'hull white', 'schwartz', 'nelson siegel', 'american monte carlo']
    skills['general_quant'] = ['linear regression', 'logistic regression', 'optimization', 'kalman filter', 'probability', 'time series', 'garch', 'arma', 'arima', 'forecast', 'clustering', 'knn', 'data analysis', 'simulated annealing', 'genetic algorithm', 'bayesian inference', 'mle', 'maximum likelihood estimation', 'expectation maximization', 'em']
    skills['machine_learning'] = ['deep learning', 'machine learning', 'image processing', 'nlp', 'neural networks', 'convolutional', 'cnn', 'lstm', 'attention', 'svm', 'ensemble', 'recognition', 'random forest', 'gradient boosting', 'bootstrap']

    skills['science_major'] = ['mathematics', 'physics', 'engineering', 'computer science', 'statistics']
    skills['frontend_programming'] = ['react', 'angular', 'js', 'javascript', 'jquery', 'node', 'html', 'css', 'coffeescript', 'html5', 'ux', 'ui', 'd3', 'phantom']
    skills['specialmentions'] = ['olympiad', 'kaggle', 'hackathon', 'prize', 'competition', 'summa', 'first class', 'distinction', 'startups', 'scholarship', 'master', 'phd', 'award']  # gpa above 4.5
    skills['markets_set'] = ['commodities', 'gold', 'oil', 'electricity', 'hedge fund', 'trading', 'fund', 'currency', 'commodity', 'currency', 'consulting', 'consultant', 'yield curve', 'investment', 'settlement', 'petroleum', 'bloomberg', 'wind', 'economics', 'gas', 'game theory', 'ipo', 'pricing', 'asset management', 'trader', 'private equity', 'fundamental', 'canslim', 'power bi', 'portfolio']
    skills['trading_houses'] = ['Gunvor', 'Trafigura', 'Glencore', 'Vitol', 'Mercuria', 'Cargill', 'ADM', 'Archer Daniels Midland', 'Bunge', 'Louis Dreyfus', 'Castleton']
    for name, words in skills.items():
        word_set = set()
        for word in words:
            word_set = word_set.union(word.split(' '))
        word_set = [word.lower() for word in word_set]
        skills[name] = word_set

    results = {}
    results['backend'] = {
        'backend_programming':  3,
        'optional_backend':     1,
        'framework':            3,
        'database':             2,
        'devops':               3,
        'science_major':        1,
        'quant_programming':    1,
    }
    results['quant'] = {
        'backend_programming':  1,
        'science_major':        3,
        'quant_programming':    2,
        'mathematical_finance': 3,
        'general_quant':        3,
        'machine_learning':     3,
        'specialmentions':      3,
    }
    results['data'] = {
        'backend_programming':  3,
        'frontend_programming': 2,
        'optional_backend':     1,
        'framework':            1,
        'database':             1,
    }
    results['frontend'] = {
        'backend_programming':  1,
        'frontend_programming': 2,
    }
    results['markets'] = {
        'markets_set':          2,
        'backend_programming':  1,
        'trading_houses':       1,
        'general_quant':        1,
        'science_major':        1,
    }
    ret = results[category]
    ret2 = {}
    for cat, weight in ret.items():
        ret2.update(dict(zip(skills[cat], [weight for _ in range(len(skills[cat]))])))
    return ret2


resume_folder = './data'

# all_cvs = convert_all_cvs(resume_folder)
# all_cvs = pickle.load(open(os.path.join(resume_folder, 'all_cvs.pkl'), 'rb'))
# words_by_cv = create_words_by_cv(all_cvs, resume_folder)
words_by_cv = pickle.load(open(os.path.join(resume_folder, 'words_by_cv.pkl'), 'rb'))
# all_words = set()
# for name, words in words_by_cv.items():
#     all_words = all_words.union(words)
# all_words = list(all_words)
# all_words.sort()
# all_words = [word for word in all_words if len(word) > 1]
# vocab = Vocabulary(all_words, similarity_cutoff=0.5)
# vocab_data = vocab.save()
# pickle.dump(vocab_data, open(os.path.join(resume_folder, 'vocab.pkl'), 'wb'))
#

# vocab = Vocabulary()
# vocab_data = pickle.load(open(os.path.join(resume_folder, 'vocab.pkl'), 'rb'))
# vocab.load(vocab_data)


# all_names = list(words_by_cv.keys())
# all_names.sort()

# nn = len(all_names)
# trans_matrix = np.identity(nn)
# cutoff = 5
# for i in range(nn):
#     print(i, all_names[i])
#     for j in range(nn):
#         sims = vocab.get_similarities(words_by_cv[all_names[i]], words_by_cv[all_names[j]], cutoff)
#         trans_matrix[i, j] = np.mean(sims)
#

# exceptions = []
# for i in range(trans_matrix.shape[0]):
#     if np.isnan(trans_matrix[i]).any():
#         print(i, all_names[i])
#         exceptions.append(i)
#
# all_names = [all_names[i] for i in range(len(all_names)) if i not in exceptions]
# trans_matrix = np.delete(trans_matrix, exceptions, axis=0)
# trans_matrix = np.delete(trans_matrix, exceptions, axis=1)
#
# markov_trans_matrix = trans_matrix.T / np.sum(trans_matrix, axis=1)
# trans_matrix_data = {
#     'names': all_names,
#     'matrix': markov_trans_matrix
# }
# pickle.dump(trans_matrix_data, open(os.path.join(resume_folder, 'trans_matrix.pkl'), 'wb'))


# matrix_info = pickle.load(open(os.path.join(resume_folder, 'trans_matrix.pkl'), 'rb'))
# all_names = matrix_info['names']
# markov_trans_matrix = matrix_info['matrix']
#
# # # Markov Approach
# # eigenvalues, eigenvectors = np.linalg.eig(markov_trans_matrix)
# # ieig = np.where(np.abs(np.absolute(eigenvalues)-1) < 1e-10)[0][0]
# # prob_vector = np.absolute(eigenvectors[:, 0])
# # prob_vector /= np.sum(prob_vector)
#
# # PageRank Approach
# d = 0.85
# pagerank_matrix = markov_trans_matrix.copy()
# for i in range(pagerank_matrix.shape[0]):
#     pagerank_matrix[i, i] = 0.0
# pagerank_matrix = pagerank_matrix / np.sum(pagerank_matrix, axis=0)
# M = np.identity(pagerank_matrix.shape[0]) - pagerank_matrix * d
# prob_vector = np.dot(np.linalg.inv(M), (1.0 - d) / M.shape[0] * np.ones(M.shape[0]))
#
# results = dict(zip(all_names, prob_vector))
# # for name, value in results.items():
# #     print(name + ',' + str(value))

keywords = {
    'math':         0.8,
    'machine':      0.8,
    'learning':     0.8,
    'kaggle':       1.0,
    'competition':  0.5,
    'award':        0.5,
    'winner':       0.5,
    'python':       1.0,
    'master':       1.0,
    'statistics':   0.8,
    'finance':      0.6,
    'cnn':          0.5,
    'lstm':         0.5
}

keywords = generate_keywords('backend')
# scores, sorted_names = rank_cvs(words_by_cv, keywords, results)
scores, sorted_names = rank_cvs2(words_by_cv, keywords)
for name in sorted_names:
    print(name + ',' + str(scores[name]) + ',' + str(sorted_names.index(name)))


# for name, words in words_by_cv.items():
#     if len(words) <= 10:
#         print(name, words)
#
#
text = 'Apple is the BEST stock in the world'
# doc = nlp(text)
# for token in doc:
#     print(token, token.vector_norm, token.orth_, token.norm_, token.lemma_, token.is_oov)
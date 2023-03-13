import csv
import math
from collections import Counter #pour chaque valeur d'un attribut il compte le nombre de repetition
import pandas as pd  #pour la manipulation des listes de listes plus aisement
import numpy as np
def read_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        attributes = next(reader)  # Lire les noms des attributs
        attributs=attributes[:len(attributes)-1]
        attribut_classe=attributes[-1]
        classes = set()  # Ensemble des classes
        data = []  # Les données
        for ligne in reader:
            class_label = ligne[-1]  # La classe est la dernière colonne
            classes.add(class_label)
            valeurs_classe=list(classes)
            example = []
            for i in range(len(attributes)):
                example.append(ligne[i])
            data.append(example)
    return attributs,attribut_classe, valeurs_classe, data


def load_donnes(filename):
    

# Ouverture du fichier CSV
    import csv

# Ouverture du fichier CSV
    with open(filename, 'r') as fichier_csv:
        # Lecture du fichier CSV avec la fonction csv.reader()
        lecteur_csv = csv.reader(fichier_csv)
        
        # Récupération de la première ligne contenant les clés
        cles = next(lecteur_csv)
        
        # Initialisation du dictionnaire
        donnees_dict = {}
        
        # Parcours de chaque ligne de données
        for ligne in lecteur_csv:
            # Parcours des valeurs de chaque ligne de données
            for j, valeur in enumerate(ligne):
                # Vérification si la clé existe déjà dans le dictionnaire
                if cles[j] in donnees_dict:
                    # Ajout de la valeur à la liste de valeurs existante
                    donnees_dict[cles[j]].append(valeur)
                else:
                    # Création d'une nouvelle liste de valeurs pour la clé
                    donnees_dict[cles[j]] = [valeur]
                
    return donnees_dict


    
    
def load_values(filename):
     with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        
        attribute_names = next(reader)
         # à partir de la deuxième ligne on a  les valeurs possibles pour chaque attribut
        attribute_values = {}
        for attribute in attribute_names:
            attribute_values[attribute] = []
    
        for row in reader:
            for i in range(len(attribute_names)):
                
                attribute_values[attribute_names[i]].append(row[i])
        return attribute_values

attributs,attribut_classe,valeurs_classe,data=read_data('golf.csv')
valeurs_attributs=load_values('golf.csv')
donnees1=load_values('golf.csv')
#transforme le dictionnaire en une liste de liste avec pour chaque colonne le nom de l'attribut et les lignes sont numerotées
donnees=pd.DataFrame(donnees1)


#la donnee en prediction
def load_values_pred(filename):
     with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        
        attribute_names = next(reader)
         # à partir de la deuxième ligne on a  les valeurs possibles pour chaque attribut
        attribute_values = {}
        for attribute in attribute_names:
            attribute_values[attribute] = []
    
        for row in reader:
            for i in range(len(attribute_names)):
                
                attribute_values[attribute_names[i]].append(row[i])
        return attribute_values
    
#donnees2=load_values_pred('soybean-pred.csv')
#nouvelles_donnees=pd.DataFrame(donnees2)

nouvelles_donnees=pd.DataFrame({'outlook':['sunny','sunny'],'temp':['hot','hot'],'humidity':['high','high'],'wind':['false','true'],'play':[None,None]})


"""def matrice_de_confusion(data, tree,target):
        # Initialisation de la matrice avec des zero
        classes = list(set(data[target]))
        num_classes = len(classes)
        class_map = {classes[i]: i for i in range(num_classes)}
        matrice_confusion = [[0 for j in range(num_classes)] for i in range(num_classes)]    
        # Calculer la matrice de confusion en parcourant l'arbre pour chaque exemple dans les données
        for _, example in data.iterrows():
            true_class = class_map[example[target]]
            predicted_class = class_map[tree.predict(pd.DataFrame([example]))[0]]
            matrice_confusion[true_class][predicted_class] += 1
            
        return matrice_confusion



def compute_confusion_matrix_pred(data_pred, tree,target):
    # Initialize the confusion matrix to zeros
    classes = list(set(data_pred[target]))
    num_classes = len(classes)
    class_map = {classes[i]: i for i in range(num_classes)}
    confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
    
    # Compute the confusion matrix by traversing the tree for each example in the data
    for _, example in data_pred.iterrows():
        true_class =class_map[ example[target]]
        predicted_class = class_map[tree.predict(pd.DataFrame([example]))[0]]
        confusion_matrix[true_class][predicted_class] += 1
        print(confusion_matrix)
    return confusion_matrix"""

class DecisionTree:
    def __init__(self, data, attributes, target):
        self.data = data
        self.attributes = attributes
        self.target = target
        self.classes = np.unique(data[target])
        self.tree = self.build_tree(data, attributes, target)
#c'est notre I(p,n)
    def entropy(self, data):   #c'est  I(p,n)
        n = len(data)
        counter = Counter(data[self.target])
        return sum([-1 * (counter[c] / n) * math.log(counter[c] / n, 2) for c in counter])

    def gain(self, data, attribute):
        n = len(data)
        values = data[attribute]
        
        entropy_total = self.entropy(data)
        entropy_values=0
        for v in values:
            
            subset = data[data[attribute] == v]
            fraction = len(subset) / n
            entropy_values += fraction * self.entropy(subset)
        #print(entropy_values)
        """entropy_values = sum([(len(data[data[attribute] == v]) / n) * self.entropy(data[data[attribute] == v]) for v in values])"""
        return entropy_total - entropy_values

    def build_tree(self, data, attributes, target):
        counter = Counter(data[target])
        # Condition 1: The subset is empty, return the most common value of the target attribute in the original data
        if len(data) == 0:
            return counter.most_common(1)[0][0]
        # Condition 2: All examples in the subset have the same value for the target attribute, return this value
        elif len(counter) == 1:
            return list(counter.keys())[0]
        # Condition 3: No attributes left to split on, return the most common value of the target attribute in the current subset
        elif len(attributes) == 0:
            return counter.most_common(1)[0][0]
        # Otherwise, choose the attribute with the highest information gain
        else:
            best_attribute = max(attributes, key=lambda a: self.gain(data, a))
            
            tree = {best_attribute: {}}
            for value in set(data[best_attribute]):
                subtree = self.build_tree(data[data[best_attribute] == value], 
                                          [a for a in attributes if a != best_attribute], 
                                          target)
                tree[best_attribute][value] = subtree
            return tree

    def predict(self, data):
        def predict_example(example, tree):
            if type(tree) != dict:
                return tree
            else:
                attribute = list(tree.keys())[0]
                if example[attribute] not in tree[attribute]:
                    return None
                subtree = tree[attribute][example[attribute]]
                return predict_example(example, subtree)
        return [predict_example(row, self.tree) for _, row in data.iterrows()]
    
    def confusion_matrix(self, data, predicted_classes):
        actual_classes = data[self.target]
        matrix = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                matrix[i, j] = sum((actual_classes == self.classes[i]) & (predicted_classes == self.classes[j]))
        return pd.DataFrame(matrix, columns=self.classes, index=self.classes)

    def evaluate(self, train_data, test_data):
        # Train the model on the training data
        self.tree = self.build_tree(train_data, self.attributes, self.target)



if __name__=="__main__":
    arbre=DecisionTree(donnees,attributs,attribut_classe)
    arbre.build_tree(donnees,attributs,attribut_classe)
    predicted_classes=arbre.predict(donnees)
    print(predicted_classes)
    conf_matrix=arbre.confusion_matrix(donnees,predicted_classes)
    print(conf_matrix)
    # Calculer la matrice de confusion pour les données d'apprentissage
    """Mapp = matrice_de_confusion(donnees, arbre,attribut_classe)
    Mpred = compute_confusion_matrix_pred(nouvelles_donnees, arbre,attribut_classe)"""
    
    """arbre.predict(nouvelles_donnees)"""
    
    
    
    """import pandas as pd

    data = pd.DataFrame({'age': [23, 34, 45, 27, 56, 32, 46, 36, 52, 28],
                     'salaire': [40000, 50000, 60000, 32000, 75000, 42000, 67000, 55000, 80000, 35000],
                     'education': ['bac', 'bac', 'master', 'licence', 'master', 'bac', 'master', 'licence', 'doctorat', 'licence'],
                     'statut': ['chômeur', 'employé', 'employé', 'chômeur', 'employé', 'chômeur', 'employé', 'employé', 'employé', 'chômeur']})
                     
    tree = DecisionTree(data, ['age', 'salaire', 'education'], 'statut')"""
    
    

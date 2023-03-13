import csv #pour la lecture des fichier csv
import math #pour le log
from collections import Counter #pour chaque valeur d'un attribut il compte le nombre de repetition
import pandas as pd  #pour la manipulation des listes de listes plus aisement
import numpy as np #pour la manipulation de matrice
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

nouvelles_donnees=pd.DataFrame({'outlook':['sunny','overcast','sunny'],'temp':['hot','hot','mild'],
                                'humidity':['high','high','normal'],'wind':['false','true','false'],
                                'play':[None, None,None]})




class DecisionTree:
    def __init__(self, data, attributes, target):
        self.data = data
        self.attributes = attributes
        self.target = target
        self.tree = self.build_tree(data, attributes, target)
        self.classes = np.unique(data[target])
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
            sous_ensemble = data[data[attribute] == v]
            fraction = len(sous_ensemble) / n
            entropy_values += fraction * self.entropy(sous_ensemble)
       
        
        return entropy_total - entropy_values

    def build_tree(self, data, attributes, target):
        counter = Counter(data[target])
        # Condition 1 : le sous-ensemble est vide, renvoie la valeur la plus courante de l'attribut cible dans les données d'origine
        if len(data) == 0:
            return counter.most_common(1)[0][0]
        # Condition 2 : Tous les exemples du sous-ensemble ont la même valeur pour l'attribut cible, renvoyez cette valeur
        elif len(counter) == 1:
            return list(counter.keys())[0]
        # Condition 3: No attributes left to split on, return the most common value of the target attribute in the current subset
        elif len(attributes) == 0:
            return counter.most_common(1)[0][0]
        # Condition 3 : Aucun attribut à fractionner, renvoie la valeur la plus courante de l'attribut cible dans le sous-ensemble actuel
        else:
            meilleur_attribut = max(attributes, key=lambda a: self.gain(data, a))
            
            tree = {meilleur_attribut: {}}
            for value in set(data[meilleur_attribut]):
                subtree = self.build_tree(data[data[meilleur_attribut] == value], 
                                          [a for a in attributes if a !=meilleur_attribut], 
                                          target)
                tree[meilleur_attribut][value] = subtree
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
    
    def matrice_confusion(self, data, predicted_classes):
        actual_classes =data[self.target]
        #initialisation de la matrice  de taille c*c
        matrice = np.zeros((len(self.classes), len(self.classes)))
        
        
        if list(predicted_classes)!=list(nouvelles_donnees['play']):
            print('dd')
            for i in range(len(self.classes)):
                for j in range(len(self.classes)):
                    matrice[i, j] = sum((actual_classes == self.classes[i]) & (predicted_classes == self.classes[j]))
        else :
            predicted_classes=arbre.predict(nouvelles_donnees)
            print('ciao')
            for i in range(len(self.classes)):
                for j in range(len(self.classes)):
                    matrice[i, j] = sum((actual_classes == self.classes[i]) & (predicted_classes == self.classes[j]))
        
        return pd.DataFrame(matrice, columns=self.classes, index=self.classes)

    


if __name__=="__main__":
    arbre=DecisionTree(donnees,attributs,attribut_classe)
    arbre.build_tree(donnees,attributs,attribut_classe)
    predicted_classes=pd.Series(arbre.predict(donnees))
    Mapp=arbre.matrice_confusion(donnees,predicted_classes)
    print(Mapp)
    # Calculer la matrice de confusion pour les données d'apprentissage
    #Mapp = matrice_de_confusion(donnees, arbre,attribut_classe)
    Mpred =arbre.matrice_confusion(nouvelles_donnees,arbre.predict(nouvelles_donnees))
    print(Mpred)
    # pour predire les classes
    print(arbre.predict(nouvelles_donnees))
    
    
    
    

    
    
    

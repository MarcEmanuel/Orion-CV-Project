import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor


class KvasirSEGDataLoader:
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.images_path = self.data_path / "images"
        self.masks_path = self.data_path / "masks"
        
    def load_data(self, max_samples=None):
        
        images = []
        masks = []
        filenames = []
        
        
        image_files = list(self.images_path.glob("*.jpg"))
        
        if max_samples:
            image_files = image_files[:max_samples]
            
        print(f"Carregando {len(image_files)} amostras do Kvasir-SEG...")
        
        for img_file in image_files:
           
            mask_file = self.masks_path / img_file.name
            
            if mask_file.exists():
                
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    
                    image = cv2.resize(image, (256, 256))
                    mask = cv2.resize(mask, (256, 256))
                    
                   
                    _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                    
                    images.append(image)
                    masks.append(mask_binary)
                    filenames.append(img_file.name)
        
        print(f"Dataset carregado: {len(images)} imagens")
        return images, masks, filenames
    
    def visualize_samples(self, images, masks, n_samples=6):
       
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
        
        indices = np.random.choice(len(images), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
           
            axes[0, i].imshow(images[idx], cmap='gray')
            axes[0, i].set_title(f'Imagem {idx}')
            axes[0, i].axis('off')
            
            
            axes[1, i].imshow(masks[idx], cmap='gray')
            axes[1, i].set_title(f'Máscara {idx}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

class KvasirFeatureExtractor:
    #Extrator de características para pólipos do Kvasir-SEG
    
    def extract_haralick_features(self, image):
   
        
        distances = [1, 2]  
        angles = np.deg2rad([0, 45, 90, 135])  
        
        try:
            glcm = graycomatrix(
                image.astype(np.uint8),
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True
            )
            
            properties = ['contrast', 'dissimilarity', 'homogeneity', 
                         'ASM', 'energy', 'correlation']
            
            features = {}
            for prop in properties:
                values = graycoprops(glcm, prop)
                features[f'haralick_{prop}_mean'] = np.mean(values)
                features[f'haralick_{prop}_std'] = np.std(values)
            
            return features
            
        except Exception as e:
            print(f"Erro no cálculo de Haralick: {e}")
            return {f'haralick_{prop}': 0 for prop in ['contrast', 'dissimilarity', 
                   'homogeneity', 'ASM', 'energy', 'correlation']}
    
    def extract_gabor_features(self, image):
        """Características de filtros de Gabor"""
        self.gabor_frequencies = [0.05, 0.1, 0.15, 0.25]
        self.gabor_angles = [0, 45, 90, 135]
        features = {}

        image_gabor = image.astype(np.float32) / 255.0
        
        for freq in self.gabor_frequencies:
            for angle in self.gabor_angles:
                real, imag = gabor(image_gabor, frequency=freq, theta=np.deg2rad(angle))
                
                # Magnitude
                magnitude = np.sqrt(real**2 + imag**2)
                
                features[f'gabor_f{freq}_a{angle}_mean'] = np.mean(magnitude)
                features[f'gabor_f{freq}_a{angle}_std'] = np.std(magnitude)
                features[f'gabor_f{freq}_a{angle}_energy'] = np.sum(magnitude**2)
        
        return features
    
    
    
    
    def extract_all_features(self, image, mask):
        """Extrai todas as características de uma imagem de pólipo"""
        features = {}
        
        # Características de Haralick (textura)
        features.update(self.extract_haralick_features(image))

        features.update(self.extract_gabor_features(image))

       
        
     
        return features
    

class KvasirPolypClassifier:
   
    
    def __init__(self):
        self.feature_extractor = KvasirFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight="balanced"),
        }
        self.best_model = None
    
    def create_binary_labels(self, masks):
        """Cria labels binários baseados no tamanho do pólipo"""
        labels = []
        
        for mask in masks:
            polyp_area = np.sum(mask)
            # Classificar como 'small' ou 'large' baseado na área
    
            if polyp_area < 7000:  # pixels
                labels.append('small_polyp')
            else:
                labels.append('large_polyp')
        
        return np.array(labels)
    
    def extract_features_from_dataset(self, images, masks):
        """Extrai características de todas as imagens"""
        features_list = []
        
        print("Extraindo características do Kvasir-SEG...")
        for i, (image, mask) in enumerate(zip(images, masks)):
            if i % 100 == 0:
                print(f"Processado: {i}/{len(images)}")
                
            features = self.feature_extractor.extract_all_features(image, mask)
            features_list.append(features)
        
        print("Extração concluída!")
        return pd.DataFrame(features_list)
    
    def train_and_evaluate(self, X, y):
        """Treina e avalia modelos"""
        # Remover NaN e infinitos
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTreinando {name}...")
            
            # Treinar modelo
            model.fit(X_train_scaled, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_labels': y_test,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"Acurácia: {accuracy:.3f}")
            print(f"Relatório de classificação:")
            print(results[name]['classification_report'])
        
        # Selecionar melhor modelo
        best_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_name]['model']
        print(f"\nMelhor modelo: {best_name}")
        
        return results
    
    def plot_results(self, results, X):
        """Visualiza resultados"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Comparação de acurácia
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]
        
        ax1 = axes[0]
        bars = ax1.bar(names, accuracies, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Comparação de Modelos - Kvasir-SEG')
        ax1.set_ylim([0, 1])
        
        
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
       
        best_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_name]
        
        ax2 = axes[1]
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title(f'Matriz de Confusão - {best_name}')
        ax2.set_xlabel('Predito')
        ax2.set_ylabel('Real')
        
        
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = X.columns.tolist()
            
            
            indices = np.argsort(feature_importance)[::-1][:10]
            top_features = [feature_names[i] for i in indices]  # ← Nomes das features
            top_importances = feature_importance[indices]
            
            ax3 = axes[2]
            bars = ax3.barh(range(10), top_importances)  # ← Usar barh (horizontal)
            ax3.set_yticks(range(10))
            ax3.set_yticklabels(top_features)  # ← Mostrar nomes das features!
            ax3.set_title('Top 10 Features Mais Importantes')
            ax3.set_xlabel('Importância')
            ax3.set_ylabel('Características')

            ax3.invert_yaxis()

            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        plt.show()

def main_kvasir_analysis(data_path, max_samples=200):
    
    
    print("-" * 50)
    
    
    loader = KvasirSEGDataLoader(data_path)
    images, masks, filenames = loader.load_data(max_samples=max_samples)
    
    if len(images) == 0:
        print("Nenhuma imagem encontrada! Verifique o caminho do dataset.")
        return
    
    
    print("Visualizando amostras...")
    loader.visualize_samples(images, masks, n_samples=6)
    
    
    classifier = KvasirPolypClassifier()
    
   
    print("Extraindo características...")
    X = classifier.extract_features_from_dataset(images, masks)
    
    
    y = classifier.create_binary_labels(masks)
    
    print(f"Dataset processado:")
    print(f"- {len(X)} amostras")
    print(f"- {len(X.columns)} características")
    print(f"- Classes: {np.unique(y)}")
    print(f"- Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    
    results = classifier.train_and_evaluate(X, y)
    
    
    classifier.plot_results(results, X)
    
    return classifier, results, X, y


if __name__ == "__main__":

    
    KVASIR_PATH = "C:/Users/marco/Orion/desafio/Kvasir-SEG"
    
    classifier, results, features_df, labels = main_kvasir_analysis(data_path=KVASIR_PATH, max_samples=None)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
   



  
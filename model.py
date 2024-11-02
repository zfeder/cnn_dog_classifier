from library import *
from preprocessing import *

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.visualized_images = 0  # Contatore per limitare la visualizzazione
        
        # Primo blocco: convoluzione, ReLU, pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Secondo blocco: convoluzione, ReLU, pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Strato di dropout per prevenire overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Calcolo della dimensione per il primo strato fully connected
        self.fc_input_dim = 128 * 2 * 2  # Adattato in base all'output dopo convoluzioni e pooling
        
        # Strati fully connected per la classificazione binaria
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Primo blocco convoluzionale
        if self.visualized_images < 3:  # Visualizza solo le prime tre immagini
            self._visualize_feature_maps(x, "Input images")
        x = self.pool(torch.relu(self.conv1(x)))
        if self.visualized_images < 3:
            self._visualize_feature_maps(x, "Output conv1")
        x = self.pool(torch.relu(self.conv2(x)))
        if self.visualized_images < 3:
            self._visualize_feature_maps(x, "Output conv2")
        
        # Secondo blocco convoluzionale
        x = self.pool(torch.relu(self.conv3(x)))
        if self.visualized_images < 3:
            self._visualize_feature_maps(x, "Output conv3")
        x = self.pool(torch.relu(self.conv4(x)))
        if self.visualized_images < 3:
            self._visualize_feature_maps(x, "Output conv4")
            self.visualized_images += 1  # Incrementa il contatore dopo la visualizzazione di un'immagine
        
        # Appiattimento del tensore con calcolo automatico della dimensione
        x = x.view(x.size(0), -1)
        
        # Strati fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Applica il dropout
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def _visualize_feature_maps(self, feature_maps, title):
        num_features = feature_maps.size(1)
        fig, axes = plt.subplots(1, min(num_features, 8), figsize=(15, 5))
        for i in range(min(num_features, 8)):
            ax = axes[i]
            ax.imshow(feature_maps[0, i].detach().cpu(), cmap='viridis')
            ax.axis('off')
        plt.suptitle(title)
        plt.show()

# Inizializza la CNN per la classificazione binaria
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = BinaryCNN().to(device)

# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss per classificazione binaria
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Funzione di addestramento con controllo delle dimensioni
def train(model, loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)  # Trasferisci su GPU
            
            # Azzeramento dei gradienti
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calcolo della perdita
            loss = criterion(outputs, labels)
            
            # Backward pass e aggiornamento dei pesi
            loss.backward()
            optimizer.step()
            
            # Accumula la perdita per il monitoraggio
            running_loss += loss.item()
        
        # Stampa la perdita media per ogni epoca
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Funzione di valutazione
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calcolo dell'accuratezza
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Avvia l'addestramento
print("Start training...")
train(cnn, train_loader, criterion, optimizer, 10)
print("Finish training")

# Valutazione sul set di validazione
print("Start evaluation...")
evaluate(cnn, val_loader, criterion)
print("Finish evaluation")

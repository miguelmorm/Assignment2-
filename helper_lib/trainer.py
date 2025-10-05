from helper_lib.checkpoints import save_checkpoint
from helper_lib.evaluator import evaluate_model
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device='cpu'):
    model.to(device)
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        #  Save checkpoint if accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch+1, val_loss, val_acc)

    # Save trained model for use in FastAPI app
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Final model saved as cnn_model.pth for API prediction")

    return model

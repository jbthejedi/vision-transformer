for batch_idx, (inputs, labels) in enumerate(train_dataloader, 1):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    # Clip gradients to a maximum norm of 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ...


total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
logging.info(f"Gradient norm: {total_norm:.4f}")


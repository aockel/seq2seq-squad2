import torch


def train_loop(model, train_dl, test_dl, optimizer, criterion, scheduler, clip, teaching_ratio, device, epochs, model_out_path):
    """
    :param model: seq2seq model
    :param train_dl:
    :param test_dl:
    :param optimizer: optimizer
    :param criterion: loss function
    :param scheduler: learning rate scheduler
    :param clip:
    :param device:
    :param epochs:
    :param model_out_path:
    :return:
    """
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # model, iterator, optimizer, criterion, clip
        train_loss = train(model, train_dl, optimizer, criterion, clip, teaching_ratio, device)
        valid_loss = evaluate(model, test_dl, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_out_path)
        # execute step for learning rate scheduler based on validation loss
        scheduler.step(valid_loss)
        # print results
        print(f'\tEpoch: {epoch:3d} | \t',
              f'Train Loss: {train_loss:.3f} | \t ',
              f'Val. Loss: {valid_loss:.3f} | \t ',
              f'lr: {optimizer.param_groups[0]["lr"]}')


def train(model, iterator, optimizer, criterion, clip, teaching_ratio, device):
    model.train()

    epoch_loss = 0

    for step, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        # reset gradient
        optimizer.zero_grad()
        # run seq2seq model
        output = model(src, trg, teaching=teaching_ratio)
        # calc loss
        outputs_flatten = output[1:].view(-1, output.shape[-1])
        trg_flatten = trg[1:].view(-1)
        loss = criterion(outputs_flatten, trg_flatten)
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for step, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, teaching=0)  # turn off teacher forcing
            # calc loss
            outputs_flatten = output[1:].view(-1, output.shape[-1])
            trg_flatten = trg[1:].view(-1)
            loss = criterion(outputs_flatten, trg_flatten)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

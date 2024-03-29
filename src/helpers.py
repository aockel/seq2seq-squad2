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
        # valid_loss = evaluate(model, test_dl, criterion, device)
        # if valid_loss < best_valid_loss:
        #    best_valid_loss = valid_loss
        #    torch.save(model.state_dict(), model_out_path)
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), model_out_path)
        # execute step for learning rate scheduler based on validation loss
        scheduler.step(train_loss)
        # print results
        print(f'\tEpoch: {epoch:3d} | \t',
              f'Train Loss: {train_loss:.3f} | \t ',
              # f'Val. Loss: {valid_loss:.3f} | \t ',
              f'lr: {optimizer.param_groups[0]["lr"]}')


def train(model, iterator, optimizer, criterion, clip, teaching_ratio, device):
    model.train()

    epoch_loss = 0

    for step, (src, trg) in enumerate(iterator):
        # print(f'Source: {src.shape}')
        # print(f'Target: {trg.shape}')
        # torch.Size([batch_size, max_length])
        src = src.to(device)
        trg = trg.to(device)
        # reset gradient
        optimizer.zero_grad()
        # run seq2seq model
        output = model(src, trg, teaching=teaching_ratio)
        # calc loss
        # transpose dim 1 and 2
        outputs_permuted = output.permute(0, 2, 1)
        # make memory layout contiguous
        outputs_contig = outputs_permuted.contiguous()
        # we need a flattened vector for the output of shape (batch_size * trg_len, trg_vocab_size)
        outputs_flatten = outputs_contig.view(-1, outputs_contig.size(-1))
        # the trg tensor is flattened to a 1-dimensional tensor of shape (batch_size * trg_len,)
        trg_flatten = trg.view(-1)
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
            # transpose dim 1 and 2
            outputs_permuted = output.permute(0, 2, 1)
            # make memory layout contiguous
            outputs_contig = outputs_permuted.contiguous()
            # we need a flattened vector for the output of shape (batch_size * trg_len, trg_vocab_size)
            outputs_flatten = outputs_contig.view(-1, outputs_contig.size(-1))
            # the trg tensor is flattened to a 1-dimensional tensor of shape (batch_size * trg_len,)
            trg_flatten = trg.view(-1)
            loss = criterion(outputs_flatten, trg_flatten)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

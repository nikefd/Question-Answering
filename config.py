class Config:
    num_epochs = 30
    batch_size = 32
    train_embeddings=0
    max_gradient_norm=-1
    hidden_state_size=150
    embedding_size=300
    data_dir="data/squad"
    vocab_path="data/squad/vocab.dat"
    embed_path="data/squad/glove.trimmed.300.npz"
    dropout_val=0.25
    train_dir="src/models_lstm_basic"
    type_of_decode=1  #1-Match LSTM, 2-Vanilla LSTM, 3-AoA
#    use_match=1


    def get_paths(mode):
        question = "data/squad/%s.ids.question" %mode
        context = "data/squad/%s.ids.context" %mode
        answer = "data/squad/%s.span" %mode

        return question, context, answer

    question_train, context_train, answer_train = get_paths("train")
    question_dev ,context_dev ,answer_dev = get_paths("val")

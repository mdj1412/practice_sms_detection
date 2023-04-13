


def train(model, train_inputs, train_labels, validation_inputs, validation_labels,
            batch_size_train, batch_size_validation,
            output_dir,
            save_boundary_accuracy,
            learning_rate=1e-5,
            # XXX
            warmup_steps=50,
            # how many steps will training
            num_training_steps=2000,
            num_epochs=20,
            # XXX
            gradient_accumulation_steps=1,
            # XXX
            max_grad_norm=1.0,
            # when will you save it during the steps
            eval_period=50,
            device='mps'):
    

    # batch size = 16, step = 200 : 3200 training examples
    # 1 epoch = 4210 steps (4210 * 16 training examples)


    # optimizer와 scheduler를 가져온다.
    optimizer, scheduler = get_optimizer_and_scheduler(
        # 해당 모델 + weight 정보
        model,
        # optimizer 에서 필요한 정보
        learning_rate=learning_rate,
        # XXX
        warmup_steps=warmup_steps,
        # scheduler 에서 필요한 정보 : 몇 steps 학습 시킬 것인지
        num_training_steps=num_training_steps
    )
    strong_indices = torch.tensor(strong[batch_idx], dtype=torch.long).cuda()
    weak_indices = torch.tensor(weak[batch_idx], dtype=torch.long).cuda()
    no_indices = torch.tensor(no[batch_idx], dtype=torch.long).cuda()

    prediction = student_predict_result[batch_idx]  # 获取当前批次的预测结果

    strong_predicted_res = prediction.index_select(0, strong_indices)
    weak_predicted_res = prediction.index_select(0, weak_indices)
    no_predicted_res = prediction.index_select(0, no_indices)

    strong_above = strong_predicted_res.log().sum()
    weak_above = weak_predicted_res.log().sum()
    strong_below_1 = strong_predicted_res.flip(0).cumsum(0)
    strong_below_2 = weak_predicted_res.sum()  # weak-correlated
    strong_below_3 = no_predicted_res.sum()  # non-correlated

    strong_loss = (strong_below_1 + strong_below_2 + strong_below_3).log().sum() - strong_above
    weak_loss = (strong_below_3 - weak_above)

    tot_loss += strong_loss
    tot_loss += weak_loss

return tot_loss * self.loss_weight
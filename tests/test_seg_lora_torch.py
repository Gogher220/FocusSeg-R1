"""Run on the cluster; no checkpoint download. Optional on CPU-only tooling hosts."""
import importlib.util
import unittest

READY = all(importlib.util.find_spec(p) is not None for p in ("torch", "peft", "transformers"))


@unittest.skipUnless(READY, "PyTorch/PEFT environment required")
class SegRoutingTests(unittest.TestCase):
    def test_row_routing_keeps_original_weights(self):
        import torch
        from seg_lora.model import SegEmbedding, SegHead
        embedding = torch.nn.Embedding(7, 4).requires_grad_(False)
        head = torch.nn.Linear(4, 7, bias=False).requires_grad_(False)
        before_in, before_out = embedding.weight.clone(), head.weight.clone()
        wrapped_in = SegEmbedding(embedding, 5, torch.ones(4))
        wrapped_out = SegHead(head, 5, torch.ones(4))
        ids = torch.tensor([[1, 5, 2]])
        values = wrapped_in(ids)
        self.assertTrue(torch.equal(values[0, 1], torch.ones(4)))
        wrapped_out(values).sum().backward()
        self.assertIsNotNone(wrapped_in.seg_row.grad)
        self.assertIsNotNone(wrapped_out.seg_row.grad)
        wrapped_in.active = wrapped_out.active = False
        self.assertTrue(torch.equal(wrapped_in(ids), embedding(ids)))
        self.assertTrue(torch.equal(wrapped_out(values), head(values)))
        self.assertTrue(torch.equal(before_in, embedding.weight))
        self.assertTrue(torch.equal(before_out, head.weight))

    def test_point_loss_negative_has_gradient(self):
        import torch
        from seg_lora.model import point_losses
        pred = torch.zeros(2, 1, 8, 8, requires_grad=True)
        bce, dice = point_losses(pred, torch.zeros_like(pred), {
            "mask_points": 64, "oversample_ratio": 3, "importance_sample_ratio": .75})
        (2*bce + .5*dice).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

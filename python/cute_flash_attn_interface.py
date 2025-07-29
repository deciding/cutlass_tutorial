import torch

if __name__ == "__main__":
    from flash_attn.cute.interface import flash_attn_func
    from python.test_util import attention_ref

    Z, H, N_CTX, HEAD_DIM = 1, 2, 1024, 128
    DEVICE = torch.device('cuda:0')

    def test_op(Z, H, N_CTX, HEAD_DIM, DEVICE, causal=False, dtype=torch.bfloat16):
        torch.manual_seed(20)
        #q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        #k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        #v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        q = (torch.empty((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        k = (torch.empty((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        v = (torch.empty((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
        #sm_scale = 0.5
        #dout = torch.randn_like(q)
        ## reference implementation
        #p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        #if causal:
        #    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        #    p[:, :, M == 0] = float("-inf")
        #p = torch.softmax(p.float(), dim=-1).bfloat16()
        #ref_out = torch.matmul(p, v).bfloat16()

        ref_out, ref_attn = attention_ref(q, k, v, causal=False)

        # B N_CTX 3 H HEAD_DIM
        #qkv = torch.randn((Z, N_CTX, 3, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        #qkv = torch.stack([q, k, v], dim=2)
        #qkv = qkv.permute(0, 3, 2, 1, 4)
        #split_tensors = qkv.permute(0, 3, 2, 1, 4)
        #q, k, v = torch.unbind(split_tensors, dim=2)

        #flash_out = flash_attn_qkvpacked_func(qkv, causal=False)
        #flash_out = flash_out.permute(0, 2, 1, 3).bfloat16()

        flash_out, lse = flash_attn_func(q, k, v, causal=False)

        print(f"Output max diff: {(flash_out - ref_out).abs().max().item()}")
        print(f"Output mean diff: {(flash_out - ref_out).abs().mean().item()}")

        # compare
        assert torch.allclose(ref_out, flash_out, atol=1e-2, rtol=0)

    test_op(Z, H, N_CTX, HEAD_DIM, DEVICE)


**Paper**

Wu, W., He, L.*, Lin, W., Mao, R., & Jarvis, S. (2021). SAFA: a Semi-Asynchronous Protocol for Fast Federated Learning with Low Overhead. IEEE Transactions on Computers (TC). vol. 70, no.5, pp. 655-668.

**Env**
- Python 3.7.3
- Pytorch 1.1.0
- PySyft 0.1.2  # optional
- numpy 1.16.4

**SAFA Protocol**

- Lag-tolerant model distribution:  

              w_k(t) = W(t-1), if k in Union_(v=t-1){Mv}           // Latest clients will sync. with server
              w_k(t) = W(t-1), if k in Union_(v<t-tao){Mv}         // Deprecated clients are forced to sync. 
              w_k(t) = w_k(t-1), if k in Union_(t-tao<=v<t-1){Mv}  // Moderately straggling clients stay async.

- Local update: 

              w_k(t) = clientUpdate(k, w_k(t)), if k in M-K(t)     // post-training w_k in round t 

- Pre-aggregation Cache update: 

              w*_k(t) = w_k(t), for k in P                         // Update entries of picked clients
              w*_k(t) = W(t-1), for k in Union_(v<t-tao){Mv}       // Deprecated entries are replaced     
           
- SAFA aggregation: 

              W(t) = sum_(k in M){n_k/n * w*_k(t)}  
- Post-aggregation Cache update:

              w*_k(t+1) = w_k(t), for k in Q                       // undrafted updates bypass round t
              w*_k(t+1) = w*_k(t), for k in P                      // already updated in pre-aggregation update 
              w*_k(t+1) = w*_k(t), for k in K                      // no update for crashed clients
where _M_ is the client set, _Mv_ is the version-v client set, _P_ is picked set, _Q_ is undrafted set, _K_ is the crash set, _tao_ denotes lag tolerance.

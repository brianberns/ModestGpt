# A modest GPT

This GPT is originally based on Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), with my own modifications and enhancements, including:

* Ported to the .NET platform (using [TorchSharp](https://github.com/dotnet/TorchSharp) instead of PyTorch) and written in F#, using a functional style. I've also refactored the code to be a bit cleaner and easier to understand for a newbie (such as myself).

* A [text generator application](https://github.com/brianberns/ModestGpt/tree/master/Apps/TokenGpt) that tokenizes using a [.NET version of TikToken](https://github.com/tryAGI/Tiktoken).

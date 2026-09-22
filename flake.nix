{
  description = "DRISHYA: Automated Diabetic Retinopathy Screening Pipeline (MATLAB & PyTorch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        # Libraries required by PyTorch, OpenCV, ReportLab, and Albumentations runtime
        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.glib
          pkgs.libGL
          pkgs.libx11
          pkgs.libxext
          pkgs.libxrender
          pkgs.cudatoolkit
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python311
            pkgs.uv
            pkgs.nodejs
            pkgs.cudatoolkit
            pkgs.git
            pkgs.git-lfs
          ];

          shellHook = ''
            export PATH="$PWD/.venv/bin:$PATH"
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export MATLAB_PATH="/home/lev/.nix-profile/bin/matlab"
            echo "DRISHYA Development Environment Loaded"
            echo "Python: $(python3 --version 2>/dev/null || echo 'N/A') | Node: $(node --version 2>/dev/null || echo 'N/A') | uv: $(uv --version 2>/dev/null || echo 'N/A')"
          '';
        };
      }
    );
}

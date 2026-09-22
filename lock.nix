# lock.nix - Pinned Nixpkgs evaluation matching flake.lock for standalone builds
let
  lock = builtins.fromJSON (builtins.readFile ./flake.lock);
  nixpkgsInfo = lock.nodes.nixpkgs.locked;
  nixpkgsSrc = fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/${nixpkgsInfo.rev}.tar.gz";
    sha256 = nixpkgsInfo.narHash;
  };
in
  import nixpkgsSrc {
    config.allowUnfree = true;
    config.cudaSupport = true;
  }

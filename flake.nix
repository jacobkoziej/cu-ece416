{
  description = "The Cooper Union - ECE 416: Adaptive Algorithms";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

        python = pkgs.python3;

        python-pkgs = python.withPackages (
          python-pkgs: with python-pkgs; [
            numpy
            scipy
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              python-pkgs
            ]
            ++ (with pkgs; [
              treefmt2
            ]);
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}

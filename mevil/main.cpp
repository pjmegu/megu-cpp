#include "llvm/Support/CommandLine.h"
#include <string>
using namespace llvm;

int main(int argc, char **argv) {
  cl::ResetCommandLineParser();
  
  cl::SubCommand lint("lint", "Run lint checks");

  auto c = cl::OptionCategory("workspaces");

  cl::opt<std::string> workspace("w", cl::desc("Workspace path"), cl::Required, cl::sub(lint), cl::cat(c));
  cl::alias workspace_a("workspace", cl::desc("Workspace path"), cl::aliasopt(workspace));

  cl::ParseCommandLineOptions(argc, argv, "mevil: mokey build system\n");

  return 0;
}
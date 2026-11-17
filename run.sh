#!/usr/bin/env sh
# Runs one lab with Java's source launcher. No build step is needed, only JDK 25+.
#   ./run.sh lab1_vector/DotProduct
#   ./run.sh lab3_tinynn/MatMul 1024
set -e
[ -z "$1" ] && { echo "usage: $0 <labN_xxx/ClassName> [args...]"; exit 1; }
lab="${1%.java}"; shift
java="${JAVA_HOME:+$JAVA_HOME/bin/}java"
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  "$dir/labs/src/main/java/playground/$lab.java" "$@"

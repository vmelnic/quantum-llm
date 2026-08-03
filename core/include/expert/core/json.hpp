#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace expert::core::json {

// Parser/canonicalizer for the deterministic JSON emitted by the P1 compiler.
// Integer values are normalized; non-integral tokens preserve the compiler's
// canonical lexical representation so the runtime reproduces its content hash.
struct Value {
    struct Number {
        std::string token;
    };
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value, std::less<>>;
    using Storage = std::variant<std::nullptr_t, bool, std::int64_t,
                                 std::uint64_t, Number, std::string, Array, Object>;

    Storage data;

    Value() : data(nullptr) {}
    explicit Value(Storage value) : data(std::move(value)) {}

    [[nodiscard]] const Object& AsObject(std::string_view where) const;
    [[nodiscard]] Object& AsObject(std::string_view where);
    [[nodiscard]] const Array& AsArray(std::string_view where) const;
    [[nodiscard]] const std::string& AsString(std::string_view where) const;
    [[nodiscard]] bool AsBool(std::string_view where) const;
    [[nodiscard]] std::uint64_t AsU64(std::string_view where) const;
};

class Error final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] Value Parse(std::string_view text);
[[nodiscard]] std::string Canonicalize(const Value& value);
[[nodiscard]] std::string Canonicalize(std::string_view text);

[[nodiscard]] const Value& Required(const Value::Object& object,
                                    std::string_view key,
                                    std::string_view where);
void RequireExactKeys(const Value::Object& object,
                      std::initializer_list<std::string_view> required,
                      std::initializer_list<std::string_view> optional,
                      std::string_view where);

}  // namespace expert::core::json

// static const int accelerate = []() {
//   ios::sync_with_stdio(false);
//   cin.tie(nullptr);
//   return 0;
// }();
// /**
//  * Definition for a binary tree node.
//  * struct TreeNode {
//  *     int val;
//  *     TreeNode *left;
//  *     TreeNode *right;
//  *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//  * };
//  */
// class Solution {
// public:
//   vector<int> flipMatchVoyage(TreeNode* root, vector<int>& voyage) {

//   }

//   template <typename Iterator>
//   bool try_fix(TreeNode* root, Iterator first, Iterator last) {
//     if (!root) return !root && first - last == 0;
//     else {
//       bool left = try_fix(root->left, first+1, last);
//       if (!left) {
//         try_flip = try_flip_root(root, );
//       }
//       if (root->val == *first)
//       else return false;
//     }
//   }
// };

// class Solution {
//   // template <typename Iterator>
//   // void reduce(Iterator first, Iterator last,
//   //             Iterator d_first, Iterator d_last)
//   // {
//   //   for (; d_first != d_last, first != last; ++d_first, ++first ) {
//   //     if ()
//   //   }
//   // }
//   void reduce(string& S) {
//     auto pos = std::find(S.rbegin(), S.rend(), '.');
//     if (pos == std::string::npos) return;
//     auto leftParen = std::find(S.rbegin(), S.rend(), '(');
//     auto rightParen = std::find(S.rbegin(), S.rend(), ')');
//     if (leftParen == S.rend() || rightParen == S.rend())
//       return;

//     auto nonRepeatIter = leftParen+1;
//     auto repeatIter = rightParen+1;
//     // while non repeating part doesn't reach '.'
//     while (nonRepeatIter != pos) {
//       for (; repeatIter != leftParen &&
//              nonRepeatIter != pos &&
//              *repeatIter == *nonRepeatIter;
//            ++repeatIter, ++nonRepeatIter)
//         { }
//       // not a full match
//       if (repeatIter != leftParen) break;
//       // restart
//       else repeatIter = rightParen+1;
//     }

//     // consume all non repeat and repeat is not at start
//     // rotate to obtain the minimum form
//     if (nonRepeatIter == pos && repeatIter != rightParen+1) {
//       std::rotate(rightParen+1, repeatIter-1, leftParen);
//       // rotate to front, and remove all latter elements
//       auto mid = std::rotate(pos.base(), leftParen.base(), rightParen.base()+1);
//       S.erase(mid, S.end());
//     }
//     else {
//       auto rotatePos =
//         nonRepeatIter - std::distance(rightParen, repeatIter);
//       auto mid =
//         std::rotate(rotatePos.base(), leftParen.base(), rightParen.base()+1);
//       S.erase(mid, S.end());
//     }
//   }

// public:
//   bool isRationalEqual(string S, string T) {
//     reduce(S), reduce(T);
//     return true;
//   }
// };

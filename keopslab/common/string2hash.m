function hash=string2hash(str,type)
% This function generates a hash value from a text string
%
% hash=string2hash(str,type);
%
% inputs,
%   str : The text string, or array with text strings.
% outputs,
%   hash : The hash value, integer value between 0 and 2^32-1
%   type : Type of has 'djb2' (default) or 'sdbm'
%
% From c-code on : http://www.cse.yorku.ca/~oz/hash.html 
%
% djb2
%  this algorithm was first reported by dan bernstein many years ago 
%  in comp.lang.c
%
% sdbm
%  this algorithm was created for sdbm (a public-domain reimplementation of
%  ndbm) database library. it was found to do well in scrambling bits, 
%  causing better distribution of the keys and fewer splits. it also happens
%  to be a good general hashing function with good distribution.
%
% example,
%
%  hash=string2hash('hello world');
%  disp(hash);
%
% Function is written by D.Kroon University of Twente (June 2010)
% Adapted by b. Charlier (2018)


% From string to double array
str=double(str);
if(nargin<2), type='both'; end

switch(type)
    case 'djb2'
        hash = 5381*ones(size(str,1),1); 
        for i=1:size(str,2), 
            hash = mod(hash * 33 + str(:,i), 2^32-1); 
        end
    case 'sdbm'
        hash = zeros(size(str,1),1);
        for i=1:size(str,2), 
            hash = mod(hash * 65599 + str(:,i), 2^32-1);
        end
case 'both'
        hash1 = 5381*ones(size(str,1),1); 
        hash2 = 5381*ones(size(str,1),1); 
        for i=1:size(str,2), 
            hash1 = mod(hash1 * 33 + str(:,i), 2^32-1); 
            hash2 = mod(hash2 * 65599 + str(:,i), 2^32-1);
        end
        hash = hash1 *1e10 +hash2;

otherwise
    error('string_hash:inputs','unknown type');
end

hash = num2str(hash,'%020.0f');

end
